import os
import streamlit as st
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

# 加载中文模型
model = SentenceTransformer('distiluse-base-multilingual-cased')

st.title("与你的日记对话")

# 获取当前日期
today = datetime.now().date()

# 创建一个日期选择器
selected_date = st.date_input("Select a date", value=today)
with st.form(key='my_form', clear_on_submit=True):
    # 表单的输入字段
    note = st.text_area(label="在这里写你的日记。", height=300)
    # 提交按钮
    submit_button = st.form_submit_button(label='Save')

# 当用户点击保存按钮时
if submit_button:
    if note:
        selected_date = str(selected_date)
        # 读取日记文件
        try:
            with open("diary.json", 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # 如果文件不存在或内容无效，则初始化为空字典
            data = {}

        # 如果选择的日期已经有日记，则追加内容
        if selected_date in data.keys():
            if note not in data[selected_date]:
                data[selected_date] += f"\n\n{note}"
        else:
            # 否则，为该日期创建新的日记条目
            data[selected_date] = note
        # 将更新后的日记数据写回文件
        with open(r'diary.json', 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False)

# 让用户输入问题
question = st.text_input(label="输入你的问题:")
# 创建一个按钮，用户点击后进行提问
button = st.button("询问")

# 当用户点击提问按钮时
if button:
    if question:
        # 初始化LLM模型
        llm = Ollama(model='llama3.2')
        # 定义用于回答问题的提示模板
        template = """
你是一个十分优秀的助手，擅长帮人根据他的日记内容回答问题。请根据以下日记回答以下问题。:
Question: {question}

Diary: {text}
"""
        prompt_template = PromptTemplate(template=template, input_variables=["text", "question"])

        # 读取日记文件
        with open(r"diary.json", 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        # 初始化一个空列表来存储相似度
        similarities = []
        # 遍历日记数据，计算每个条目与问题的相似度
        question_embedding = model.encode(question)
        for date, diary in data.items():
            diary_embedding = model.encode(diary)
            similarity_score = util.pytorch_cos_sim(question_embedding, diary_embedding)
            similarities.append((similarity_score.item(), f"Date: {date}\nDiary: {diary}"))

        # 根据相似度对条目进行排序，并取前三个最相关的条目
        ordered_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

        # 初始化一个空字符串，用于存储最相关的日记条目
        text = ""
        # 将最相关的日记条目添加到文本中
        for score, sentence in ordered_chunks:
            st.markdown(f"Similarity: {score:.4f} - Sentence: {sentence}")
            text += f"{sentence}\n\n"
        # 使用LLMChain和提示模板来生成答案
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run({"text": text, "question": question})

        # 显示答案
        st.subheader("Answer:")
        st.markdown(answer)
