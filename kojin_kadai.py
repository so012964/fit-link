import os
import json
import re
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF


# ============================================================
# 0. 環境設定
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Fit Link",
    layout="wide",
)

# パスワード認証（Secretsに設定されている場合のみ）
if "APP_PASSWORD" in st.secrets:
    password = st.text_input("パスワードを入力してください", type="password")
    if password != st.secrets["APP_PASSWORD"]:
        st.warning("パスワードを知っている人だけが利用できます。")
        st.stop()

# キャプションの文字サイズを調整
st.markdown("""
<style>
    .stCaption {
        font-size: 1.0rem !important;
    }
</style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 1. モデル指定
# ============================================================

MODEL_DIAG = "gpt-5-mini"  # 抽出・判定・質問・対話
MODEL_WRITE = "gpt-5.2"    # 最終追記文（提出用）


# ============================================================
# 2. 出力制約
# ============================================================

MAX_REQUIREMENTS = 25
MAX_GAPS = 25
EVIDENCE_MAX_CHARS = 140
MAX_QUESTIONS_LACK = 10
MAX_QUESTIONS_UNKNOWN = 5
ADDENDUM_MAX_CHARS = 1200

MIN_INPUT_CHARS = 100  # 入力の最低文字数（警告用）


# ============================================================
# 3. 例外
# ============================================================

class OpenAIAppError(RuntimeError):
    pass


# ============================================================
# 4. 推論漏れ検知
# ============================================================

LEAK_PATTERNS = [
    r"推論", r"思考過程", r"chain[- ]?of[- ]?thought", r"step\s*by\s*step",
    r"おそらく", r"たぶん", r"と思われ", r"と考え", r"考えると",
    r"したがって", r"よって", r"結論",
    r"理由\s*[:：]", r"根拠\s*[:：]", r"判断\s*[:：]", r"分析\s*[:：]",
]
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


def find_leak_paths_all(obj: Any, base_path: str = "$") -> List[Tuple[str, str]]:
    leaks: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaks.extend(find_leak_paths_all(v, f"{base_path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaks.extend(find_leak_paths_all(v, f"{base_path}[{i}]"))
    elif isinstance(obj, str):
        if LEAK_REGEX.search(obj):
            leaks.append((base_path, obj[:200].replace("\n", " ")))
    return leaks


def find_leak_paths_limited(obj: Any, allow_path_prefixes: List[str]) -> List[Tuple[str, str]]:
    leaks: List[Tuple[str, str]] = []

    def _is_relevant_for_traversal(path: str) -> bool:
        return any(path.startswith(pfx) or pfx.startswith(path) for pfx in allow_path_prefixes)

    def _walk(x: Any, path: str) -> None:
        if path != "$" and not _is_relevant_for_traversal(path):
            return
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                _walk(v, f"{path}[{i}]")
        elif isinstance(x, str):
            if any(path.startswith(pfx) for pfx in allow_path_prefixes):
                if LEAK_REGEX.search(x):
                    leaks.append((path, x[:200].replace("\n", " ")))

    _walk(obj, "$")
    return leaks


# ============================================================
# 5. JSON Schema
# ============================================================

SCHEMA_REQUIREMENTS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "maxItems": MAX_REQUIREMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": 10},
                    "text": {"type": "string", "maxLength": 240},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requirements"],
    "additionalProperties": False,
}

SCHEMA_GAPS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "gaps": {
            "type": "array",
            "maxItems": MAX_GAPS,
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {"type": "string", "maxLength": 10},
                    "status": {"type": "string", "enum": ["アピール済み", "補足が必要", "記載なし"]},
                    "resume_evidence": {"type": "string", "maxLength": EVIDENCE_MAX_CHARS},
                },
                "required": ["requirement_id", "status", "resume_evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["gaps"],
    "additionalProperties": False,
}

SCHEMA_QUESTIONS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": MAX_QUESTIONS_LACK + MAX_QUESTIONS_UNKNOWN,
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {"type": "string", "maxLength": 10},
                    "question": {"type": "string", "maxLength": 220},
                },
                "required": ["requirement_id", "question"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

# 対話（STEP4）専用スキーマ：次の1問 or 素材確定（finalize）
SCHEMA_DIALOG: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {"type": "string", "maxLength": 10},
                    "fact": {"type": "string", "maxLength": 240},
                },
                "required": ["requirement_id", "fact"],
                "additionalProperties": False,
            },
        },
        "next_question": {"type": "string", "maxLength": 240},
        "done": {"type": "boolean"},
    },
    "required": ["facts", "next_question", "done"],
    "additionalProperties": False,
}



# ============================================================
# 6. PDF→テキスト抽出
# ============================================================

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 30) -> str:
    chunks: List[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = min(len(doc), max_pages)
        for i in range(pages):
            chunks.append(doc[i].get_text("text"))
    return "\n".join(chunks).strip()


# ============================================================
# 7. Debug logger（主画面に生JSONは出さない）
# ============================================================

DEBUG_MAX_EVENTS = 30
DEBUG_RAW_HEAD = 1200

def _safe_head(s: Optional[str], n: int = DEBUG_RAW_HEAD) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n")
    return s[:n] + ("..." if len(s) > n else "")

def log_debug_event(event_type: str, payload: Dict[str, Any]) -> None:
    if "debug_events" not in st.session_state or st.session_state["debug_events"] is None:
        st.session_state["debug_events"] = []
    st.session_state["debug_events"].append({"type": event_type, "payload": payload})
    if len(st.session_state["debug_events"]) > DEBUG_MAX_EVENTS:
        st.session_state["debug_events"] = st.session_state["debug_events"][-DEBUG_MAX_EVENTS:]


# ============================================================
# 8. OpenAI 呼び出し（schema validation + 例外の見せ方）
# ============================================================

def _extract_refusal_or_raise(response: Any) -> None:
    try:
        if response.output and response.output[0].content:
            first = response.output[0].content[0]
            if isinstance(first, dict) and first.get("type") == "refusal":
                raise OpenAIAppError(f"リクエストが拒否されました: {first.get('refusal','')}")
            if getattr(first, "type", None) == "refusal":
                raise OpenAIAppError(f"リクエストが拒否されました: {getattr(first, 'refusal', '')}")
    except OpenAIAppError:
        raise
    except Exception:
        return


def get_output_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    outs: List[str] = []
    for item in (getattr(response, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            t = None
            if isinstance(c, dict):
                if c.get("type") in ("output_text", "text"):
                    t = c.get("text")
            else:
                if getattr(c, "type", None) in ("output_text", "text"):
                    t = getattr(c, "text", None)
            if isinstance(t, str) and t.strip():
                outs.append(t.strip())

    return "\n".join(outs).strip()


RE_REQUIREMENT_ID = re.compile(r"^R\d+$")

def validate_requirements_obj(data: Dict[str, Any]) -> None:
    reqs = data.get("requirements")
    if not isinstance(reqs, list):
        raise OpenAIAppError("requirements が配列ではありません。")

    ids: List[str] = []
    for r in reqs:
        if not isinstance(r, dict):
            raise OpenAIAppError("requirements の要素形式が不正です。")
        rid = r.get("id")
        txt = r.get("text")
        if not isinstance(rid, str) or not RE_REQUIREMENT_ID.match(rid):
            raise OpenAIAppError("requirements のID形式が不正です（R1, R2,...）。")
        if not isinstance(txt, str) or not txt.strip():
            raise OpenAIAppError("requirements のtextが空です。")
        ids.append(rid)

    if len(ids) != len(set(ids)):
        raise OpenAIAppError("requirements のIDが重複しています。")


def validate_gaps_obj(data: Dict[str, Any], requirements: Optional[Dict[str, Any]] = None) -> None:
    gaps = data.get("gaps")
    if not isinstance(gaps, list):
        raise OpenAIAppError("gaps が配列ではありません。")

    allowed = {"アピール済み", "補足が必要", "記載なし"}

    req_ids = None
    if isinstance(requirements, dict):
        reqs = requirements.get("requirements")
        if isinstance(reqs, list):
            req_ids = {r.get("id") for r in reqs if isinstance(r, dict)}

    for g in gaps:
        if not isinstance(g, dict):
            raise OpenAIAppError("gaps の要素形式が不正です。")
        rid = g.get("requirement_id")
        status = g.get("status")
        ev = g.get("resume_evidence")

        if not isinstance(rid, str) or not rid:
            raise OpenAIAppError("gaps.requirement_id が不正です。")
        if req_ids is not None and rid not in req_ids:
            raise OpenAIAppError("gaps.requirement_id が requirements と一致しません。")

        if status not in allowed:
            raise OpenAIAppError("gaps.status が不正です。")
        if not isinstance(ev, str):
            raise OpenAIAppError("gaps.resume_evidence が不正です。")


def validate_questions_obj(data: Dict[str, Any], requirements: Optional[Dict[str, Any]] = None) -> None:
    qs = data.get("questions")
    if not isinstance(qs, list):
        raise OpenAIAppError("questions が配列ではありません。")

    req_ids = None
    if isinstance(requirements, dict):
        reqs = requirements.get("requirements")
        if isinstance(reqs, list):
            req_ids = {r.get("id") for r in reqs if isinstance(r, dict)}

    for q in qs:
        if not isinstance(q, dict):
            raise OpenAIAppError("questions の要素形式が不正です。")
        rid = q.get("requirement_id")
        question = q.get("question")
        if not isinstance(rid, str) or not rid:
            raise OpenAIAppError("questions.requirement_id が不正です。")
        if req_ids is not None and rid not in req_ids:
            raise OpenAIAppError("questions.requirement_id が requirements と一致しません。")
        if not isinstance(question, str) or not question.strip():
            raise OpenAIAppError("questions.question が空です。")


def validate_dialog_obj(data: Dict[str, Any], requirements: Optional[Dict[str, Any]] = None) -> None:
    facts = data.get("facts")
    next_q = data.get("next_question")
    done = data.get("done")

    if not isinstance(facts, list):
        raise OpenAIAppError("dialog.facts が配列ではありません。")
    if not isinstance(next_q, str):
        raise OpenAIAppError("dialog.next_question が文字列ではありません。")
    if not isinstance(done, bool):
        raise OpenAIAppError("dialog.done が boolean ではありません。")

    # facts要素チェック
    req_ids = None
    if isinstance(requirements, dict):
        reqs = requirements.get("requirements")
        if isinstance(reqs, list):
            req_ids = {r.get("id") for r in reqs if isinstance(r, dict)}

    for f in facts:
        if not isinstance(f, dict):
            raise OpenAIAppError("dialog.facts の要素形式が不正です。")
        rid = f.get("requirement_id")
        fact = f.get("fact")
        if not isinstance(rid, str) or not rid:
            raise OpenAIAppError("dialog.facts.requirement_id が不正です。")
        if req_ids is not None and rid not in req_ids:
            raise OpenAIAppError("dialog.facts.requirement_id が requirements と一致しません。")
        if not isinstance(fact, str) or not fact.strip():
            raise OpenAIAppError("dialog.facts.fact が空です。")



def call_openai_json_schema(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    schema: Dict[str, Any],
    max_output_tokens: int,
    retries: int = 2,
    leak_check: bool = True,
    leak_allow_path_prefixes: Optional[List[str]] = None,
    context_requirements: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    - 主画面に生JSONを表示しない方針
    - 失敗時は短いユーザー向け例外にまとめ、詳細は debug_events に格納
    """
    last_raw: Optional[str] = None
    last_leaks: List[Tuple[str, str]] = []

    for attempt in range(retries + 1):
        if attempt == 0:
            effective_user_prompt = user_prompt
        else:
            leak_lines = "\n".join([f"- {p}: {s}" for p, s in last_leaks[:10]])
            effective_user_prompt = f"""
先ほどの出力に「推論過程・理由説明」が混入しています。修正してください。

禁止:
- 推論/理由/判断/分析/結論 などの説明文
- 「〜と考えます」「おそらく」などの推測

混入箇所:
{leak_lines}

【修正対象】
{last_raw}

【元の指示】
{user_prompt}
""".strip()

        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": effective_user_prompt.strip()},
                ],
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as e:
            log_debug_event("api_call_error", {
                "schema_name": schema_name, "attempt": attempt, "model": model, "error": repr(e),
            })
            raise OpenAIAppError("API呼び出し中にエラーが発生しました。もう一度お試しください。") from e

        try:
            _extract_refusal_or_raise(response)
        except OpenAIAppError as e:
            log_debug_event("refusal", {
                "schema_name": schema_name, "attempt": attempt, "model": model, "error": str(e),
            })
            raise

        if response.status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            log_debug_event("incomplete", {
                "schema_name": schema_name, "attempt": attempt, "model": model, "reason": reason,
            })
            raise OpenAIAppError("出力が不完全でした。入力を短くするか、もう一度お試しください。")

        if response.status != "completed":
            log_debug_event("unexpected_status", {
                "schema_name": schema_name, "attempt": attempt, "model": model, "status": response.status,
            })
            raise OpenAIAppError("処理が完了しませんでした。もう一度お試しください。")

        raw = get_output_text(response)
        last_raw = raw

        if not raw:
            log_debug_event("empty_output", {
                "schema_name": schema_name, "attempt": attempt, "model": model,
            })
            raise OpenAIAppError("出力が空でした。もう一度お試しください。")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log_debug_event("json_decode_error", {
                "schema_name": schema_name, "attempt": attempt, "model": model,
                "error": repr(e), "raw_head": _safe_head(raw),
            })
            raise OpenAIAppError("出力形式の解析に失敗しました。もう一度お試しください。") from e

        if leak_check:
            if leak_allow_path_prefixes:
                leaks = find_leak_paths_limited(data, leak_allow_path_prefixes)
            else:
                leaks = find_leak_paths_all(data)

            if leaks:
                last_leaks = leaks
                log_debug_event("leak_detected", {
                    "schema_name": schema_name, "attempt": attempt, "model": model,
                    "leaks": leaks[:10], "raw_head": _safe_head(raw),
                })
                continue

        # 軽量バリデーション（strict schemaの保険）
        try:
            if schema_name == "requirements_schema":
                validate_requirements_obj(data)
            elif schema_name == "gaps_schema":
                validate_gaps_obj(data, requirements=context_requirements)
            elif schema_name == "questions_schema":
                validate_questions_obj(data, requirements=context_requirements)
            elif schema_name == "dialog_schema":
                validate_dialog_obj(data)
        except OpenAIAppError as ve:
            log_debug_event("validation_error", {
                "schema_name": schema_name, "attempt": attempt, "model": model,
                "error": str(ve), "raw_head": _safe_head(raw),
            })
            raise OpenAIAppError("出力内容の整合チェックでエラーになりました。もう一度お試しください。") from ve

        return data

    log_debug_event("retries_exhausted", {
        "schema_name": schema_name,
        "model": model,
        "last_raw_head": _safe_head(last_raw),
        "last_leaks": last_leaks[:10],
    })
    raise OpenAIAppError("出力の整形に失敗しました。もう一度お試しください。")


def call_openai_text(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
) -> str:
    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            max_output_tokens=max_output_tokens,
        )
    except Exception as e:
        log_debug_event("api_call_error_text", {"model": model, "error": repr(e)})
        raise OpenAIAppError("文章生成の呼び出しでエラーが発生しました。もう一度お試しください。") from e

    _extract_refusal_or_raise(response)

    if response.status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        log_debug_event("incomplete_text", {"model": model, "reason": reason})
        raise OpenAIAppError("文章生成が不完全でした。もう一度お試しください。")

    if response.status != "completed":
        log_debug_event("unexpected_status_text", {"model": model, "status": response.status})
        raise OpenAIAppError("文章生成が完了しませんでした。もう一度お試しください。")

    out = get_output_text(response)
    if not out:
        log_debug_event("empty_output_text", {"model": model})
        raise OpenAIAppError("文章生成の出力が空でした。もう一度お試しください。")

    return out


# ============================================================
# 9. プロンプト
# ============================================================

SYSTEM_REQUIREMENTS = """
あなたは求人票から「応募者に求められるスキル・経験」を抽出する専門家です。

重要な区別:
- 「入社後に担当する業務内容」は抽出対象外
- 「応募時点で持っていてほしいスキル・経験・資格」のみを抽出

例:
- ❌「〇〇大学でIR戦略を策定する」→ これは入社後の業務内容なので除外
- ✅「IR戦略策定の経験」→ これは求められるスキル
- ❌「△△システムを導入・運用する」→ これは入社後の業務内容なので除外
- ✅「システム導入・運用の経験」→ これは求められるスキル

厳守:
- 特定の組織名（応募先企業・大学名）での経験は求めない（汎用的なスキルとして抽出）
- 推論・理由説明は出力しない
- JSONのみ出力
""".strip()

SYSTEM_GAP = """
あなたは求人要件と職務経歴書の一致状況を判定する専門家です。

厳守:
- status は「アピール済み」「補足が必要」「記載なし」のいずれか
- resume_evidence は職務経歴書からの短い引用または要約のみ
- 推論・理由説明は出力しない
- JSONのみ出力
""".strip()

SYSTEM_INTERVIEW = """
あなたは職務経歴書の補強をサポートする専門家です。
応募者が「書き漏らしている経験や実績」を思い出せるよう、質問を作成します。

厳守:
- 質問は1項目につき最大1問
- 尋問調ではなく、サポート調で
- 推論・理由説明は出力しない
- JSONのみ出力
""".strip()

# STEP4：対話専用ロール
SYSTEM_DIALOG_EDITOR = """
あなたは職務経歴書の補強を支援する編集者です。
求人要件（1件）に対し、ユーザーが書き足せる「事実」を思い出せるように短い質問をします。

厳守:
- 推測・誇張は禁止
- 1回の出力は「次の1問」か「確定した内容（箇条書き）」のどちらか
- 質問するとき：質問は1つだけ、短く具体的に
- 確定するとき：facts に書き足す内容（箇条書き、事実のみ）を入れる（数字・期間・役割・成果があると良い）
- 推論・理由説明は出力しない
- JSONのみ出力
""".strip()

SYSTEM_WRITER = """
あなたは提出用の職務経歴書に書き足す文章を作成する編集者です。

あなたの役割:
- ユーザーが入力した素材（事実）をもとに、職務経歴書にふさわしい文章に仕上げる
- 誤字脱字を修正する
- 文章として自然な表現・体裁に整える
- 箇条書きの場合は、表現を統一し、読みやすく整形する

厳守:
- ユーザーが回答した事実の意味を変えない
- 推測・誇張は禁止（事実を膨らませない）
- 不明点は（要確認：xxx）で残す
""".strip()


# ============================================================
# 10. ユーティリティ
# ============================================================

NON_ESSENTIAL_PATTERNS = [
    r"提出", r"応募", r"申込", r"申請", r"エントリ",
    r"締切", r"〆切", r"期限",
    r"郵送", r"メール", r"フォーム", r"URL", r"Web", r"オンライン",
    r"履歴書", r"職務経歴書", r"添付", r"PDF", r"書類",
]
NON_ESSENTIAL_REGEX = re.compile("|".join(NON_ESSENTIAL_PATTERNS), re.IGNORECASE)

def default_include_flag(text: str) -> bool:
    if not isinstance(text, str):
        return True
    return not bool(NON_ESSENTIAL_REGEX.search(text))

def build_selected_requirements_from_editor(edited_df: pd.DataFrame) -> Dict[str, Any]:
    selected = edited_df[edited_df["診断に含める"] == True].copy()  # noqa: E712
    return {"requirements": [{"id": r["ID"], "text": r["内容"]} for _, r in selected.iterrows()]}

def uniq_questions_by_requirement_id(questions_obj: Dict[str, Any]) -> Dict[str, Any]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for q in (questions_obj.get("questions", []) or []):
        rid = q.get("requirement_id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        uniq.append(q)
    return {"questions": uniq}

def build_need_review_requirement_ids(gaps: Dict[str, Any]) -> List[str]:
    """
    対話対象（補足が必要/記載なし）の requirement_id だけ抽出。
    """
    ids: List[str] = []
    for g in (gaps.get("gaps", []) or []):
        if g.get("status") in ("補足が必要", "記載なし"):
            rid = g.get("requirement_id")
            if isinstance(rid, str) and rid:
                ids.append(rid)

    # 重複排除（順序維持）
    seen = set()
    uniq = []
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        uniq.append(rid)
    return uniq


# ============================================================
# 11. 診断ロジック
# ============================================================

def extract_requirements(job_text: str) -> Dict[str, Any]:
    return call_openai_json_schema(
        model=MODEL_DIAG,
        system_prompt=SYSTEM_REQUIREMENTS,
        user_prompt=f"""
求人票から「求められるスキル・経験」を抽出してください。

【求人票】
{job_text}

制約:
- 重要度順に最大{MAX_REQUIREMENTS}件まで
- 重複は統合
- IDは R1, R2, ... と連番
- JSONのみ
""".strip(),
        schema_name="requirements_schema",
        schema=SCHEMA_REQUIREMENTS,
        max_output_tokens=4096,
        retries=2,
        leak_check=False,
    )

def classify_gaps(requirements: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
    return call_openai_json_schema(
        model=MODEL_DIAG,
        system_prompt=SYSTEM_GAP,
        user_prompt=f"""
以下の「求められるスキル・経験」と職務経歴書を照合してください。

【求められるスキル・経験（JSON）】
{json.dumps(requirements, ensure_ascii=False)}

【職務経歴書】
{resume_text}

制約:
- status は「アピール済み」「補足が必要」「記載なし」のいずれか
- resume_evidence は最大{EVIDENCE_MAX_CHARS}文字
- 根拠が無い場合は "該当する記載なし" と入れる
- JSONのみ

判定基準:
- アピール済み = 職務経歴書に十分な記載がある
- 補足が必要 = 記載はあるが情報が不足している
- 記載なし = 関連する記載が見つからない
""".strip(),
        schema_name="gaps_schema",
        schema=SCHEMA_GAPS,
        max_output_tokens=8192,
        retries=2,
        leak_check=True,
        leak_allow_path_prefixes=["$.gaps"],
        context_requirements=requirements,
    )

def make_questions(gaps: Dict[str, Any], requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return call_openai_json_schema(
        model=MODEL_DIAG,
        system_prompt=SYSTEM_INTERVIEW,
        user_prompt=f"""
以下の「補足が必要」「記載なし」の項目について、思い出しを促す質問を作成してください。

【照合結果（JSON）】
{json.dumps(gaps, ensure_ascii=False)}

目的:
- 応募者が「書き漏らしている経験や実績」を思い出せるようにする
- 職務経歴書に書き足せる内容を引き出す

制約:
- 「補足が必要」は最大{MAX_QUESTIONS_LACK}問
- 「記載なし」は最大{MAX_QUESTIONS_UNKNOWN}問
- 1項目につき最大1問
- 指示語（この業務/この経験 等）を避ける
- JSONのみ
""".strip(),
        schema_name="questions_schema",
        schema=SCHEMA_QUESTIONS,
        max_output_tokens=4096,
        retries=2,
        leak_check=True,
        leak_allow_path_prefixes=["$.questions"],
        context_requirements=requirements,
    )

def dialog_refine_one_requirement(
    *,
    requirement_id: str,
    requirement_text: str,
    resume_text: str,
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    STEP4：1要件について対話を1ステップ進める
    chat_history: [{"role":"user"/"assistant", "content":"..."}]
    """
    user_prompt = f"""
対象要件:
- id: {requirement_id}
- text: {requirement_text}

職務経歴書（参考）:
{resume_text}

これまでの対話:
{json.dumps(chat_history, ensure_ascii=False)}

上の情報を踏まえ、次のアクションを選んでください:
- まだ情報が足りないなら done=false で次の1問を next_question に入れる
- 十分なら done=true で facts（書き足す内容）を確定する
""".strip()

    return call_openai_json_schema(
        model=MODEL_DIAG,
        system_prompt=SYSTEM_DIALOG_EDITOR,
        user_prompt=user_prompt,
        schema_name="dialog_schema",
        schema=SCHEMA_DIALOG,
        max_output_tokens=2048,
        retries=1,
        leak_check=True,
        leak_allow_path_prefixes=["$"],
    )

def write_addendum(resume_text: str, requirements: Dict[str, Any], answers: Dict[str, str]) -> str:
    return call_openai_text(
        model=MODEL_WRITE,
        system_prompt=SYSTEM_WRITER,
        user_prompt=f"""
以下を踏まえて、職務経歴書に書き足す文章を作成してください。

【対象の求められるスキル・経験（JSON）】
{json.dumps(requirements, ensure_ascii=False)}

【元の職務経歴書】
{resume_text}

【書き足す内容（JSON）】
{json.dumps(answers, ensure_ascii=False)}

制約:
- 最大{ADDENDUM_MAX_CHARS}文字を目安に
- 回答にない事実は追加しない
- 不明点は（要確認：xxx）で残す
- 出力は書き足す文章のみ

推奨構成:
- 書き足し候補（箇条書き 3〜8点）
- 必要なら短い補足文
""".strip(),
        max_output_tokens=8192,
    )


# ============================================================
# 12. UIヘルパー
# ============================================================

def render_progress_stepper(current_step: int):
    steps = [
        "① 入力",
        "② スキル整理",
        "③ 経歴と比較",
        "④ 内容を整理",
        "⑤ 文章を作成",
    ]
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        step_num = i + 1
        with col:
            if step_num < current_step:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; background-color:#d4edda; "
                    f"border-radius:8px; border:2px solid #28a745;'><b>✅ {step_name}</b></div>",
                    unsafe_allow_html=True
                )
            elif step_num == current_step:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; background-color:#cce5ff; "
                    f"border-radius:8px; border:2px solid #007bff;'><b>▶ {step_name}</b></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:center; padding:10px; background-color:#f8f9fa; "
                    f"border-radius:8px; border:1px solid #dee2e6; color:#6c757d;'>{step_name}</div>",
                    unsafe_allow_html=True
                )

def get_status_style(status: str) -> Tuple[str, str]:
    if status == "アピール済み":
        return "#d4edda", "✅"
    elif status == "補足が必要":
        return "#fff3cd", "⚠️"
    elif status == "記載なし":
        return "#f8d7da", "❌"
    return "#ffffff", ""

def render_gaps_summary(gaps: Dict[str, Any]):
    gap_list = gaps.get("gaps", [])
    counts = {"アピール済み": 0, "補足が必要": 0, "記載なし": 0}
    for g in gap_list:
        s = g.get("status", "")
        if s in counts:
            counts[s] += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ アピール済み", counts["アピール済み"])
    with col2:
        st.metric("⚠️ 補足が必要", counts["補足が必要"])
    with col3:
        st.metric("❌ 記載なし", counts["記載なし"])

def render_gaps_detail(gaps: Dict[str, Any], requirements: Dict[str, Any]):
    gap_list = gaps.get("gaps", [])
    req_map = {r["id"]: r["text"] for r in requirements.get("requirements", [])}

    for g in gap_list:
        rid = g.get("requirement_id", "")
        status = g.get("status", "")
        evidence = g.get("resume_evidence", "")
        req_text = req_map.get(rid, "（不明）")

        bg_color, emoji = get_status_style(status)
        st.markdown(f"""
        <div style="background-color:{bg_color}; padding:12px; border-radius:8px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span><b>{rid}</b>: {req_text[:60]}{'...' if len(req_text) > 60 else ''}</span>
                <span style="font-size:1.1em;">{emoji} {status}</span>
            </div>
            <div style="margin-top:6px; color:#555; font-size:0.9em;">
                根拠: {evidence if evidence else '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_questions_cards(questions: Dict[str, Any], requirements: Dict[str, Any]):
    q_list = questions.get("questions", [])
    req_map = {r["id"]: r["text"] for r in requirements.get("requirements", [])}

    for q in q_list:
        rid = q.get("requirement_id", "")
        question = q.get("question", "")
        req_text = req_map.get(rid, "")
        st.markdown(f"""
        <div style="background-color:#e7f3ff; padding:12px; border-radius:8px;
                    margin-bottom:10px; border-left:4px solid #007bff;">
            <div style="font-size:0.85em; color:#666; margin-bottom:6px;">
                📌 {rid}: {req_text[:60]}{'...' if len(req_text) > 60 else ''}
            </div>
            <div style="font-size:1.0em;">
                💬 {question}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_requirement_card(rid: str, req_text: str, subtitle: str = ""):
    sub = f"<div style='font-size:0.85em; color:#666; margin-bottom:8px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style="background-color:#f8f9fa; padding:16px; border-radius:8px; margin-bottom:8px; border-left:4px solid #007bff;">
        {sub}
        <div style="font-weight:600; margin-bottom:8px;">📌 求められるスキル（{rid}）</div>
        <div style="color:#333;">{req_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 13. メインUI（状態管理）
# ============================================================

st.title("Fit Link")
st.markdown("""
<p style="font-size: 1.1rem; color: #666; margin-top: -10px;">
—求められていることと、積み重ねてきたことを、結び直す。
</p>
""", unsafe_allow_html=True)

APP_STATE_KEYS = [
    "job_text_snapshot",
    "resume_text_snapshot",
    "requirements_raw",
    "requirements_selected",
    "gaps",
    "questions",
    "need_review_ids",
    "dialog_by_requirement",
    "dialog_queue",
    "dialog_index",
    "addendum_materials",
    "addendum_selected",       # ★ 書き足す対象として選択されたもの
    "addendum_text",
    "current_step",
    "debug_events",
]

for k in APP_STATE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = None

if st.session_state["current_step"] is None:
    st.session_state["current_step"] = 1

def reset_app_state():
    for k in APP_STATE_KEYS:
        st.session_state[k] = None
    st.session_state["current_step"] = 1

def get_requirements_for_use() -> Optional[Dict[str, Any]]:
    return st.session_state.get("requirements_selected") or st.session_state.get("requirements_raw")

# 進捗ステッパー
st.markdown("---")
render_progress_stepper(st.session_state["current_step"])
st.markdown("---")


# ============================================================
# STEP 1: 入力
# ============================================================

if st.session_state["current_step"] == 1:
    st.header("① 求人票と職務経歴書を入力")

    col_job, col_resume = st.columns(2)

    with col_job:
        st.subheader("求人票")
        job_mode = st.radio("入力方法", ["テキスト貼り付け", "PDFアップロード"], horizontal=True, key="job_mode")

        if job_mode == "テキスト貼り付け":
            job_text = st.text_area("求人票の内容をコピー＆ペースト", height=280, key="job_text_input",
                                    placeholder="求人票のテキストを貼り付けてください...")
        else:
            job_pdf = st.file_uploader("PDFファイルを選択", type=["pdf"], key="job_pdf")
            if job_pdf:
                with st.spinner("読み込み中..."):
                    job_text = extract_text_from_pdf(job_pdf.getvalue())
                if job_text:
                    job_text = st.text_area("抽出結果（必要に応じて編集）", value=job_text, height=250, key="job_text_from_pdf")
                else:
                    st.warning("テキストを抽出できませんでした。手動で入力してください。")
                    job_text = ""
            else:
                job_text = ""

        job_len = len(job_text) if job_text else 0
        if job_len > 0 and job_len < MIN_INPUT_CHARS:
            st.warning(f"入力が短いです（{job_len}文字）。求人票全体を入力してください。")
        elif job_len > 0:
            st.caption(f"{job_len}文字")

    with col_resume:
        st.subheader("職務経歴書")
        resume_mode = st.radio("入力方法", ["テキスト貼り付け", "PDFアップロード"], horizontal=True, key="resume_mode")

        if resume_mode == "テキスト貼り付け":
            resume_text = st.text_area("職務経歴書の内容をコピー＆ペースト", height=280, key="resume_text_input",
                                       placeholder="職務経歴書のテキストを貼り付けてください...")
        else:
            resume_pdf = st.file_uploader("PDFファイルを選択", type=["pdf"], key="resume_pdf")
            if resume_pdf:
                with st.spinner("読み込み中..."):
                    resume_text = extract_text_from_pdf(resume_pdf.getvalue())
                if resume_text:
                    resume_text = st.text_area("抽出結果（必要に応じて編集）", value=resume_text, height=250, key="resume_text_from_pdf")
                else:
                    st.warning("テキストを抽出できませんでした。手動で入力してください。")
                    resume_text = ""
            else:
                resume_text = ""

        resume_len = len(resume_text) if resume_text else 0
        if resume_len > 0 and resume_len < MIN_INPUT_CHARS:
            st.warning(f"入力が短いです（{resume_len}文字）。職務経歴書全体を入力してください。")
        elif resume_len > 0:
            st.caption(f"{resume_len}文字")

    can_proceed = (job_text and len(job_text) >= MIN_INPUT_CHARS and resume_text and len(resume_text) >= MIN_INPUT_CHARS)

    if st.button("次へ進む →", use_container_width=True, type="primary", disabled=not can_proceed):
        st.session_state["job_text_snapshot"] = job_text
        st.session_state["resume_text_snapshot"] = resume_text
        st.session_state["current_step"] = 2

        # 以降をリセット
        st.session_state["requirements_raw"] = None
        st.session_state["requirements_selected"] = None
        st.session_state["gaps"] = None
        st.session_state["questions"] = None
        st.session_state["need_review_ids"] = None
        st.session_state["dialog_by_requirement"] = None
        st.session_state["dialog_queue"] = None
        st.session_state["dialog_index"] = None
        st.session_state["addendum_materials"] = None
        st.session_state["addendum_selected"] = None
        st.session_state["addendum_text"] = None
        st.rerun()

    if not can_proceed and (job_text or resume_text):
        st.info("求人票・職務経歴書の両方を十分な長さで入力すると次へ進めます。")

else:
    # 入力内容の折りたたみ
    with st.expander("入力済みの内容を確認", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("求人票（先頭300文字）")
            job_preview = (st.session_state.get("job_text_snapshot") or "")[:300]
            st.text(job_preview + ("..." if len(st.session_state.get("job_text_snapshot") or "") > 300 else ""))
        with col2:
            st.caption("職務経歴書（先頭300文字）")
            resume_preview = (st.session_state.get("resume_text_snapshot") or "")[:300]
            st.text(resume_preview + ("..." if len(st.session_state.get("resume_text_snapshot") or "") > 300 else ""))


# ============================================================
# STEP 2: スキル整理（要件抽出）
# ============================================================

if st.session_state["current_step"] == 2:
    st.header("② 求められるスキル・経験を整理")

    if st.session_state.get("requirements_raw") is None:
        st.markdown("求人票から「求められるスキル・経験」を自動で抽出します。")

        if st.button("抽出を開始", use_container_width=True, type="primary"):
            try:
                with st.status("求人票を分析しています...", expanded=True) as status:
                    st.write("求人票の内容を読み込んでいます...")
                    st.write("求められるスキル・経験を抽出中...")
                    st.caption("（通常20〜40秒かかります）")

                    req = extract_requirements(st.session_state["job_text_snapshot"])

                    status.update(label="抽出が完了しました", state="complete")

                st.session_state["requirements_raw"] = req
                st.rerun()

            except OpenAIAppError as e:
                st.error(f"エラーが発生しました: {e}")
            except Exception as e:
                log_debug_event("unexpected_exception_step2", {"error": repr(e)})
                st.error(f"予期しないエラーが発生しました: {e}")

    else:
        req_rows = st.session_state["requirements_raw"].get("requirements", [])
        st.success(f"{len(req_rows)}件のスキル・経験を抽出しました")

        st.markdown("#### 診断に含める項目を選択")
        st.caption("応募方法など診断に不要な項目は外してください")

        df_req = pd.DataFrame(req_rows)
        if "include" not in df_req.columns:
            df_req["include"] = df_req["text"].apply(default_include_flag)

        df_req = df_req[["include", "id", "text"]].rename(columns={"include": "診断に含める", "id": "ID", "text": "内容"})

        edited = st.data_editor(
            df_req,
            use_container_width=True,
            hide_index=True,
            column_config={
                "診断に含める": st.column_config.CheckboxColumn("含める", width="small"),
                "ID": st.column_config.TextColumn("ID", disabled=True, width="small"),
                "内容": st.column_config.TextColumn("内容", disabled=True),
            },
            key="requirements_editor",
        )

        selected_req = build_selected_requirements_from_editor(edited)
        selected_count = len(selected_req["requirements"])
        st.info(f"選択中: {selected_count}件 / 全{len(req_rows)}件")

        col_back, col_next = st.columns([1, 2])
        with col_back:
            if st.button("← 入力に戻る", use_container_width=True):
                st.session_state["current_step"] = 1
                st.session_state["requirements_raw"] = None
                st.session_state["requirements_selected"] = None
                st.rerun()

        with col_next:
            if st.button("次へ進む →", use_container_width=True, type="primary", disabled=(selected_count == 0)):
                st.session_state["requirements_selected"] = selected_req
                st.session_state["current_step"] = 3
                st.session_state["gaps"] = None
                st.session_state["questions"] = None
                st.session_state["need_review_ids"] = None
                st.session_state["dialog_by_requirement"] = None
                st.session_state["dialog_queue"] = None
                st.session_state["dialog_index"] = None
                st.session_state["addendum_materials"] = None
                st.session_state["addendum_selected"] = None
                st.session_state["addendum_text"] = None
                st.rerun()

elif st.session_state["current_step"] > 2:
    req_selected = get_requirements_for_use()
    if req_selected:
        with st.expander(f"整理済み: {len(req_selected.get('requirements', []))}件", expanded=False):
            for r in req_selected.get("requirements", []):
                st.markdown(f"- **{r['id']}**: {r['text']}")


# ============================================================
# STEP 3: 経歴と比較（ギャップ判定 + 質問）
# ============================================================

if st.session_state["current_step"] == 3:
    st.header("③ あなたの経歴と比較")

    req_for_gap = get_requirements_for_use()

    if st.session_state.get("gaps") is None:
        st.markdown("職務経歴書と照らし合わせて、アピールできている点・補足が必要な点を判定します。")

        if st.button("比較を開始", use_container_width=True, type="primary"):
            try:
                with st.status("職務経歴書を分析しています...", expanded=True) as status:
                    st.write("職務経歴書の内容を読み込んでいます...")
                    st.write("求められるスキルと照合中...")
                    st.caption("（通常30秒〜1分かかります）")

                    gaps = classify_gaps(req_for_gap, st.session_state["resume_text_snapshot"])

                    status.update(label="比較が完了しました", state="complete")

                st.session_state["gaps"] = gaps
                st.session_state["need_review_ids"] = build_need_review_requirement_ids(gaps)
                st.rerun()

            except OpenAIAppError as e:
                st.error(f"エラーが発生しました: {e}")
            except Exception as e:
                log_debug_event("unexpected_exception_step3_compare", {"error": repr(e)})
                st.error(f"予期しないエラーが発生しました: {e}")

        if st.button("← スキル整理に戻る", use_container_width=True):
            st.session_state["current_step"] = 2
            st.rerun()

    else:
        st.markdown("#### 比較結果（サマリー）")
        render_gaps_summary(st.session_state["gaps"])
        st.markdown("---")
        st.markdown("#### 詳細")
        render_gaps_detail(st.session_state["gaps"], req_for_gap)

        need_ids = st.session_state.get("need_review_ids") or []
        st.markdown("---")

        if len(need_ids) == 0:
            st.success("補足が必要／記載なしの項目がないため、書き足しは不要そうです。")
            col_back, col_next = st.columns([1, 2])
            with col_back:
                if st.button("← スキル整理に戻る", use_container_width=True):
                    st.session_state["current_step"] = 2
                    st.rerun()
            with col_next:
                if st.button("⑤へ進む（文章作成）→", use_container_width=True, type="primary"):
                    st.session_state["addendum_materials"] = {}
                    st.session_state["current_step"] = 5
                    st.rerun()
        else:
            st.info(f"「補足が必要」「記載なし」の項目が {len(need_ids)} 件あります。次のステップで内容を整理しましょう。")

            col_back, col_next = st.columns([1, 2])
            with col_back:
                if st.button("← スキル整理に戻る", use_container_width=True):
                    st.session_state["current_step"] = 2
                    st.session_state["gaps"] = None
                    st.session_state["questions"] = None
                    st.session_state["need_review_ids"] = None
                    st.rerun()

            with col_next:
                if st.button("④へ進む（内容を整理）→", use_container_width=True, type="primary"):
                    st.session_state["current_step"] = 4
                    st.rerun()

elif st.session_state["current_step"] > 3 and st.session_state.get("gaps"):
    gap_list = st.session_state["gaps"].get("gaps", [])
    counts = {"アピール済み": 0, "補足が必要": 0, "記載なし": 0}
    for g in gap_list:
        s = g.get("status", "")
        if s in counts:
            counts[s] += 1
    summary = f"✅{counts['アピール済み']} ⚠️{counts['補足が必要']} ❌{counts['記載なし']}"
    with st.expander(f"比較結果: {summary}", expanded=False):
        render_gaps_detail(st.session_state["gaps"], get_requirements_for_use())


# ============================================================
# STEP 4: 書き足す内容を整理（対話）
# ============================================================

if st.session_state["current_step"] == 4:
    st.header("④ 書き足す内容を整理する")
    st.caption("質問に答えながら、職務経歴書に書き足せる経験・実績を整理します。対象は「補足が必要」「記載なし」の項目のみです。")

    req_for_use = get_requirements_for_use()
    req_map = {r["id"]: r["text"] for r in (req_for_use or {}).get("requirements", [])}

    need_ids = st.session_state.get("need_review_ids") or []
    if not need_ids:
        st.info("対象がありません。⑤へ進みます。")
        st.session_state["addendum_materials"] = {}
        st.session_state["current_step"] = 5
        st.rerun()

    # 初期化（最初の表示時だけ）
    if st.session_state.get("dialog_queue") is None:
        st.session_state["dialog_queue"] = need_ids
        st.session_state["dialog_index"] = 0
        st.session_state["dialog_by_requirement"] = {}
        st.session_state["addendum_materials"] = {}

    queue: List[str] = st.session_state["dialog_queue"] or []
    idx: int = int(st.session_state.get("dialog_index") or 0)

    # 全て完了したら⑤へ
    if idx >= len(queue):
        st.success("書き足す内容の整理が完了しました。⑤へ進みます。")
        st.session_state["current_step"] = 5
        st.rerun()

    current_rid = queue[idx]
    current_req_text = req_map.get(current_rid, "")

    # セッション上の対話状態を取得
    dialog_state = st.session_state["dialog_by_requirement"].get(current_rid)
    if dialog_state is None:
        dialog_state = {
            "history": [],
            "turns": 0,
            "finalized": False,
            "facts": "",
        }
        st.session_state["dialog_by_requirement"][current_rid] = dialog_state

    # 上部：対象要件カード
    render_requirement_card(
        current_rid,
        current_req_text or "（要件テキストが見つかりません）",
        subtitle=f"進捗: {idx+1} / {len(queue)}"
    )

    st.markdown("#### 対話")
    st.caption("コツ：数字（件数/率/期間）、役割（担当範囲）、成果（改善・削減・達成）を書けると強いです。")

    # チャット履歴表示
    for m in dialog_state["history"]:
        with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
            st.markdown(m["content"])

    # finalize済みなら次へ
    if dialog_state.get("finalized"):
        st.success("この要件は整理完了です。")
        with st.expander("整理した内容（編集可）", expanded=False):
            edited_facts = st.text_area(
                "整理した内容",
                value=dialog_state.get("facts", ""),
                height=140,
                key=f"facts_edit_{current_rid}",
            )
            if isinstance(edited_facts, str):
                dialog_state["facts"] = edited_facts.strip()
                st.session_state["addendum_materials"][current_rid] = dialog_state["facts"]

        col_skip, col_next = st.columns([1, 2])
        with col_skip:
            if st.button("この要件をやり直す", use_container_width=True):
                st.session_state["dialog_by_requirement"][current_rid] = {
                    "history": [], "turns": 0, "finalized": False, "facts": ""
                }
                st.session_state["addendum_materials"].pop(current_rid, None)
                st.rerun()

        with col_next:
            if st.button("次の要件へ →", use_container_width=True, type="primary"):
                st.session_state["dialog_index"] = idx + 1
                st.rerun()

    else:
        MAX_TURNS_PER_REQ = 4

        # まだアシスタントからの最初の問いがない場合は、自動で1回生成して提示
        if len(dialog_state["history"]) == 0:
            try:
                with st.status("質問を準備しています...", expanded=False) as status:
                    out = dialog_refine_one_requirement(
                        requirement_id=current_rid,
                        requirement_text=current_req_text,
                        resume_text=st.session_state.get("resume_text_snapshot") or "",
                        chat_history=[],
                    )
                    status.update(label="準備完了", state="complete")

                dialog_state["history"].append({"role": "assistant", "content": out["next_question"]})

                if out["done"]:
                    facts_list = out.get("facts") or []
                    facts = "\n".join([f"- {f.get('fact', '')}" for f in facts_list if isinstance(f, dict) and f.get('fact')])
                    dialog_state["finalized"] = True
                    dialog_state["facts"] = facts
                    st.session_state["addendum_materials"][current_rid] = facts

                st.rerun()

            except OpenAIAppError as e:
                st.error(f"エラーが発生しました: {e}")
            except Exception as e:
                log_debug_event("unexpected_exception_step4_init", {"error": repr(e)})
                st.error(f"予期しないエラーが発生しました: {e}")

        # 入力欄（text_areaで高さを確保）
        user_input = st.text_area(
            "この要件について、思い当たる事実（実績・役割・成果など）を書いてください",
            height=100,
            key=f"dialog_input_{current_rid}_{dialog_state['turns']}",
            placeholder="例：2022年から2年間、○○のリーダーとして△△を担当し、□□を達成しました"
        )
        st.caption("💡 入力が終わったら「送信」ボタン、または Ctrl+Enter で送信できます")
        
        col_submit, col_skip, col_back = st.columns([2, 1, 1])
        with col_submit:
            submit_clicked = st.button("送信 →", use_container_width=True, type="primary", disabled=not user_input.strip())
        with col_skip:
            skip_clicked = st.button("スキップ", use_container_width=True)
        with col_back:
            back_clicked = st.button("← 戻る", use_container_width=True)

        if back_clicked:
            st.session_state["current_step"] = 3
            st.rerun()
        
        if skip_clicked:
            dialog_state["finalized"] = True
            dialog_state["facts"] = ""
            st.session_state["addendum_materials"][current_rid] = ""
            st.session_state["dialog_index"] = idx + 1
            st.rerun()

        if submit_clicked and user_input.strip():
            dialog_state["history"].append({"role": "user", "content": user_input})
            dialog_state["turns"] = int(dialog_state.get("turns") or 0) + 1

            try:
                with st.status("整理しています...", expanded=False) as status:
                    out = dialog_refine_one_requirement(
                        requirement_id=current_rid,
                        requirement_text=current_req_text,
                        resume_text=st.session_state.get("resume_text_snapshot") or "",
                        chat_history=dialog_state["history"],
                    )
                    status.update(label="更新しました", state="complete")

                dialog_state["history"].append({"role": "assistant", "content": out["next_question"]})

                if out["done"]:
                    facts_list = out.get("facts") or []
                    facts = "\n".join([f"- {f.get('fact', '')}" for f in facts_list if isinstance(f, dict) and f.get('fact')])
                    dialog_state["finalized"] = True
                    dialog_state["facts"] = facts
                    st.session_state["addendum_materials"][current_rid] = facts

                else:
                    if dialog_state["turns"] >= MAX_TURNS_PER_REQ:
                        dialog_state["finalized"] = True
                        user_lines = [m["content"] for m in dialog_state["history"] if m["role"] == "user"]
                        facts = "\n".join([f"- {line.strip()}" for line in user_lines if isinstance(line, str) and line.strip()])
                        dialog_state["facts"] = facts
                        st.session_state["addendum_materials"][current_rid] = facts

                st.rerun()

            except OpenAIAppError as e:
                st.error(f"エラーが発生しました: {e}")
            except Exception as e:
                log_debug_event("unexpected_exception_step4_turn", {"error": repr(e)})
                st.error(f"予期しないエラーが発生しました: {e}")


# ============================================================
# STEP 5: 書き足す文章を作成（提出用）
# ============================================================

if st.session_state["current_step"] == 5:
    st.header("⑤ 職務経歴書に書き足す文章を作成")

    req_for_addendum = get_requirements_for_use()
    materials: Dict[str, str] = st.session_state.get("addendum_materials") or {}
    req_map = {r["id"]: r["text"] for r in (req_for_addendum or {}).get("requirements", [])}

    # 空素材は落とす
    cleaned_materials = {k: v.strip() for k, v in materials.items() if isinstance(v, str) and v.strip()}

    if not cleaned_materials:
        st.info("書き足す内容が入力されていないため、文章は作成しません。")
        col_back, col_reset = st.columns([1, 1])
        with col_back:
            if st.button("← 内容整理に戻る", use_container_width=True):
                st.session_state["current_step"] = 4
                st.rerun()
        with col_reset:
            if st.button("最初からやり直す", use_container_width=True):
                reset_app_state()
                st.rerun()
    else:
        # ★ 書き足す対象の確認UI（チェックボックス）
        st.markdown("#### 書き足す対象を確認")
        st.caption("チェックを外した項目は、最終文章に含まれません。")

        # 初回は全て選択状態
        if st.session_state.get("addendum_selected") is None:
            st.session_state["addendum_selected"] = {rid: True for rid in cleaned_materials.keys()}

        selected_state: Dict[str, bool] = st.session_state["addendum_selected"]

        for rid, facts in cleaned_materials.items():
            req_text = req_map.get(rid, "（不明）")
            col_check, col_content = st.columns([0.08, 0.92])
            with col_check:
                is_selected = st.checkbox(
                    "",
                    value=selected_state.get(rid, True),
                    key=f"select_{rid}",
                    label_visibility="collapsed",
                )
                selected_state[rid] = is_selected
            with col_content:
                bg_color = "#f0f8ff" if is_selected else "#f5f5f5"
                text_color = "#333" if is_selected else "#999"
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:12px; border-radius:8px; margin-bottom:8px; border-left:4px solid {'#007bff' if is_selected else '#ccc'};">
                    <div style="font-weight:600; margin-bottom:6px; color:{text_color};">📌 {rid}: {req_text[:60]}{'...' if len(req_text) > 60 else ''}</div>
                    <div style="font-size:0.9em; color:{text_color}; white-space:pre-wrap;">{facts[:200]}{'...' if len(facts) > 200 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

        st.session_state["addendum_selected"] = selected_state

        # 選択された項目だけ抽出
        final_materials = {rid: facts for rid, facts in cleaned_materials.items() if selected_state.get(rid, False)}
        selected_count = len(final_materials)

        st.info(f"選択中: {selected_count}件 / 全{len(cleaned_materials)}件")

        st.markdown("---")

        # 選択された項目の内容編集
        if final_materials:
            st.markdown("#### 書き足す内容（必要なら編集）")
            st.caption("ここを直すと、最終文章にも反映されます。")

            edited_answers: Dict[str, str] = {}
            for rid, facts in final_materials.items():
                render_requirement_card(rid, req_map.get(rid, ""), subtitle="この要件に対して整理した内容")
                edited_answers[rid] = st.text_area(
                    f"内容（{rid}）",
                    value=facts,
                    height=140,
                    key=f"final_material_{rid}",
                    label_visibility="collapsed",
                ).strip()
        else:
            edited_answers = {}

        st.markdown("---")
        col_back, col_generate = st.columns([1, 2])
        with col_back:
            if st.button("← 内容整理に戻る", use_container_width=True):
                st.session_state["addendum_materials"] = {**cleaned_materials, **edited_answers}
                st.session_state["current_step"] = 4
                st.rerun()

        with col_generate:
            can_generate = bool(edited_answers) and any(v for v in edited_answers.values())
            if st.button("文章を作成する", use_container_width=True, type="primary", disabled=not can_generate):
                try:
                    final_answers = {k: v for k, v in edited_answers.items() if isinstance(v, str) and v.strip()}

                    # 選択された要件のみを渡す
                    selected_req_ids = set(final_answers.keys())
                    filtered_requirements = {
                        "requirements": [
                            r for r in (req_for_addendum or {}).get("requirements", [])
                            if r.get("id") in selected_req_ids
                        ]
                    }

                    with st.status("文章を作成しています...", expanded=True) as status:
                        st.write("整理した内容を反映して文章化しています...")
                        st.caption("（通常30秒〜1分かかります）")

                        result = write_addendum(
                            resume_text=st.session_state.get("resume_text_snapshot") or "",
                            requirements=filtered_requirements,
                            answers=final_answers,
                        )

                        status.update(label="文章が完成しました", state="complete")

                    st.session_state["addendum_text"] = result
                    st.rerun()

                except OpenAIAppError as e:
                    st.error(f"エラーが発生しました: {e}")
                except Exception as e:
                    log_debug_event("unexpected_exception_step5_generate", {"error": repr(e)})
                    st.error(f"予期しないエラーが発生しました: {e}")

        # 表示
        if st.session_state.get("addendum_text"):
            st.markdown("---")
            st.subheader("完成した文章")
            st.caption("内容を確認し、必要に応じて編集してください。このままコピーして職務経歴書に貼り付けられます。")

            edited_result = st.text_area(
                "完成した文章（編集・コピー可）",
                st.session_state["addendum_text"],
                height=300,
                key="final_addendum_edit",
                label_visibility="collapsed",
            )

            # コピーボタンの代わりにヒントを表示
            st.info("💡 上のテキストエリアを選択して Ctrl+A → Ctrl+C でコピーできます")

            # 達成感を演出（数字ベースのサマリー）
            st.markdown("---")
            st.subheader("診断完了")
            
            # 統計を計算
            req_count = len((get_requirements_for_use() or {}).get("requirements", []))
            materials_count = len([v for v in (st.session_state.get("addendum_materials") or {}).values() if v])
            
            st.markdown(f"""
            ✅ 求人票から **{req_count}** 件のスキル要件を抽出  
            ✅ うち **{materials_count}** 件について内容を整理  
            ✅ 書き足す文章を作成
            """)
            
            st.caption("この文章を職務経歴書に追加して、応募準備を進めてください。")

            st.markdown("---")
            if st.button("最初からやり直す", use_container_width=True):
                reset_app_state()
                st.rerun()


# ============================================================
# サイドバー
# ============================================================

with st.sidebar:
    st.markdown("### 進捗状況")

    step_status = {
        "① 入力": "✅" if st.session_state.get("job_text_snapshot") else "—",
        "② スキル整理": "✅" if st.session_state.get("requirements_selected") else "—",
        "③ 経歴と比較": "✅" if st.session_state.get("gaps") else "—",
        "④ 内容を整理": "✅" if (st.session_state.get("addendum_materials") and any(v for v in (st.session_state.get("addendum_materials") or {}).values())) else "—",
        "⑤ 文章を作成": "✅" if st.session_state.get("addendum_text") else "—",
    }
    for step, status in step_status.items():
        st.markdown(f"{status} {step}")

    st.markdown("---")
    if st.button("最初からやり直す", use_container_width=True):
        reset_app_state()
        st.rerun()

    st.markdown("---")
    st.caption("ヒント")
    st.caption("・求人票・経歴書は全文を入力すると精度が上がります")
    st.caption("・数字/期間/役割/成果があると強いです")
