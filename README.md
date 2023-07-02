# langchain-playground

1. 安装依赖
```sh
poetry install
```

2. 配置
```sh
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 添加 openai api key
```

3. 运行
```sh
poetry run streamlit run app.py
# or
poetry shell
streamlit run app.py
```
