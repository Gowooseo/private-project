from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

app = Flask(__name__)

# 데이터 로드
moneyDF = pd.read_csv('./data/주요국 통화의 대원화환율_25105044.csv')
moneyDF=moneyDF.drop(['통계표','단위','변환'],axis=1)
moneyDF=moneyDF.set_index('계정항목')
moneyDFT=moneyDF.T
moneyDFT_Number = moneyDFT.apply(lambda x : pd.to_numeric(x.str.replace(',' , '')))
moneyDFT_Number.reset_index(inplace=True)
moneyDFT_Number['index'] = pd.to_datetime(moneyDFT_Number['index'])


# 모델 로드
with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 스케일러 정의
cols = moneyDFT_Number.columns[1:]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(moneyDFT_Number[cols])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        date = request.form['date']
        return predict(date)
    return render_template('index.html')

def predict(date):
    # 입력 날짜를 datetime 객체로 변환
    input_date = datetime.strptime(date, '%Y-%m-%d')

    # 입력 날짜와 마지막 학습 데이터의 날짜 차이 계산
    last_date = moneyDFT_Number['index'].max()
    days_diff = (input_date - last_date).days

    # 과거 30일 데이터 준비
    input_data = scaled_data[-30:]

    # 예측 날짜만큼 반복하여 예측
    predictions = {}
    for _ in range(days_diff):
        x = np.array([input_data])
        prediction = model.predict(x)
        input_data = np.concatenate((input_data[1:], prediction.reshape(1, -1)), axis=0)

    # 예측 결과 역정규화
    prediction = scaler.inverse_transform(prediction.reshape(1, -1))

    # 예측 결과를 딕셔너리로 저장
    for i, col in enumerate(cols):
        predictions[col] = prediction[0, i]

    return render_template('results.html', predictions=predictions, date=date)

if __name__ == '__main__':
    app.run(debug=True)