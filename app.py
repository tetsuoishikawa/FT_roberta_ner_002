import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import gc

MODEL_REPO = "Tetsuo3003/ner-medical-japanese"

@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO, low_cpu_mem_usage=False)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

ner_pipeline = load_pipeline()

LABEL_COLORS = {
    "PER": "#FF6666", "ORG": "#66B2FF", "LOC": "#66FF66", "INS": "#FFCC66",
    "PRD": "#CC99FF", "EVT": "#FF99CC", "ORG-P": "#FFB266", "ORG-O": "#FFB266"
}

def mask_entities(text, entities):
    masked_text = text
    for entity in sorted(entities, key=lambda x: -len(x['word'])):
        label = entity['entity_group']
        color = LABEL_COLORS.get(label, "#CCCCCC")
        replacement = f"<span style='color: {color}; font-weight: bold;'>[{label}]</span>"
        masked_text = re.sub(re.escape(entity['word']), replacement, masked_text)
    return masked_text

st.title("🩺 日本語 医療会話 NER アプリ")

text = st.text_area("解析したいテキストを入力してください（500文字まで）:",
                    "金丸先生が松本市にある石川クリニックに通院しました。", max_chars=500)

if st.button("解析開始"):
    with st.spinner("解析中..."):
        results = ner_pipeline(text)
        masked_text = mask_entities(text, results)

        st.subheader("📝 仮名加工（マスキング）後の文章")
        st.markdown(masked_text, unsafe_allow_html=True)

        st.subheader("🔍 抽出したエンティティ一覧")
        if results:
            for entity in results:
                st.write(f"- **{entity['word']}** → {entity['entity_group']} (信頼度: {entity['score']:.2f})")
        else:
            st.info("エンティティは検出されませんでした。")

        del results
        gc.collect()
