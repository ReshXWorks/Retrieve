import streamlit as st
import requests

st.set_page_config(page_title="Retrieve: AI Document Assistant", layout="wide")

# 🔥 SAFE SESSION INIT (important)
st.session_state.setdefault("history", [])
st.session_state.setdefault("query", "")

# 🔥 CLEAN UI CSS
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}

.user-msg {
    background-color: #2b313e;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 10px;
    text-align: right;
}

.bot-msg {
    background-color: #1e222b;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 5px;
}

.meta {
    font-size: 12px;
    color: #aaa;
    margin-bottom: 10px;
}

.source-box {
    background-color: #111;
    padding: 10px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Document Assistant")

# 🔥 Sidebar Upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing document..."):
            res = requests.post(
                "http://localhost:8000/upload",
                files={"file": open("temp.pdf", "rb")}
            )

        data = res.json()
        st.success(data.get("message", "Upload complete"))

# 🔥 Chat UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for q, res in st.session_state.history:
    st.markdown(f'<div class="user-msg">🧑 {q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">🤖 {res["answer"]}</div>', unsafe_allow_html=True)

    conf = res.get("confidence", 0)
    rel = res.get("relevance", 0)

    if conf > 0.75:
        color = "🟢"
    elif conf > 0.5:
        color = "🟡"
    else:
        color = "🔴"

    st.markdown(
        f'<div class="meta">{color} Confidence: {conf} | Relevance: {rel} | Hallucination: {res["hallucination"]}</div>',
        unsafe_allow_html=True
    )

    with st.expander("Sources"):
        for s in res["sources"]:
            clean_s = s.replace("...", "").strip()
            st.markdown(f'<div class="source-box">{clean_s}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# 🔥 INPUT HANDLER (FIXED)
def handle_submit():
    query = st.session_state.query.strip()

    if not query:
        return

    with st.spinner("Thinking..."):
        res = requests.get(
            "http://localhost:8000/query",
            params={"q": query}
        )

        data = res.json()

    st.session_state.history.append((query, data))

    # ✅ SAFE CLEAR (no crash)
    st.session_state.query = ""


# 🔥 INPUT BOX
st.text_input(
    "Ask something about your document...",
    key="query"
)

st.button("Send", on_click=handle_submit)
