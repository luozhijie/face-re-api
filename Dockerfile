FROM python:3.10-bookworm

LABEL authors="lzj"

RUN pip install fastapi uvicorn numpy Cython insightface onnxruntime

RUN mkdir -p /root/.insightface/models/ \
&& wget -O /root/.insightface/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
&& unzip /root/.insightface/models/buffalo_l.zip -d /root/.insightface/models/buffalo_l/ \
&& rm -rf /root/.insightface/models/buffalo_l.zip

ADD *.py /app/

EXPOSE 80

CMD ["sh", "-c", "cd /app && uvicorn main:app --host 0.0.0.0 --port 80"]