from flask import Flask, render_template, request, jsonify
import random
import pandas as pd
import pickle
from openai import OpenAI
import os

app = Flask(__name__)

# 環境変数からOpenAI APIキーを読み込む
# openai.api_key = os.getenv("OPENAI_API_KEY")

# ラベルエンコーダーとモデルの読み込み
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('model_gf.pkl', 'rb') as f:
    model_gf = pickle.load(f)
with open('model_ga.pkl', 'rb') as f:
    model_ga = pickle.load(f)

# ユニークなチームと会場のリスト
teams = ['Liverpool', 'Manchester City', 'Arsenal', 'Aston Villa',
        'Tottenham Hotspur', 'West Ham United', 'Brighton and Hove Albion',
        'Manchester United', 'Chelsea', 'Newcastle United',
        'Wolverhampton Wanderers', 'Bournemouth', 'Fulham', 'Brentford',
        'Crystal Palace', 'Nottingham Forest', 'Everton', 'Luton Town',
        'Burnley', 'Sheffield United', 'Leicester City', 'Leeds United',
        'Southampton', 'Watford', 'Norwich City']
venues = ['Home', 'Away']

def ask_chatgpt(question, team_info, predicted_gf, predicted_ga):
    """
    ChatGPTに質問を送信し、回答を取得する。得点予測と失点予測も含める。
    """
    client = OpenAI()

    try:
        prompt = f"どちらが勝ちますか？: {question}\nチーム情報: {team_info}\n得点予測: {predicted_gf}\n失点予測: {predicted_ga}\n回答:"
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )

        response_message = stream.choices[0].message.content
        return response_message
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    selected_teams = None
    chat_response = None
    if request.method == 'POST':
        user_team = request.form.get("user_team")
        opponent_team = request.form.get('opponent_team')
        venue = request.form.get('venue')
        question = request.form.get('question', '')  # 質問があれば取得
        
        # 試合予測のデータ準備
        example_match_data = {
            'day': random.randint(0, 6),  # 週の日（0=月曜日, 6=日曜日）
            'venue': label_encoders["venue"].transform([venue]),
            'opponent': label_encoders['team'].transform([opponent_team])[0],
            # ランダムなデフォルト値
            'xg': random.uniform(0.0, 5.0),  # 予想得点（0.5から3.0の範囲でランダム）
            'xga': random.uniform(0.0, 5.0),  # 予想失点（0.5から3.0の範囲でランダム）
            'poss': random.randint(30, 70),  # ボール保持率（30%から70%の範囲でランダム）
            'attendance': random.randint(10000, 80000),  # 観客数（1万から8万の範囲でランダム）
            'sh': random.randint(5, 20),  # シュート数（5から20の範囲でランダム）
            'sot': random.randint(1, 10),  # シュートオンターゲット数（1から10の範囲でランダム）
            'dist': random.uniform(10, 35),  # シュート距離（10から30の範囲でランダム）
            'fk': random.randint(0, 10),  # フリーキック数（0から5の範囲でランダム）
            'pk': random.randint(0, 2),  # ペナルティキック数（0から2の範囲でランダム）
            'pkatt': random.randint(0, 2),  # ペナルティキック試行数（0から2の範囲でランダム）
            'team': label_encoders['team'].transform([user_team])[0]
        }
        example_match_df = pd.DataFrame([example_match_data])
        
        # 予測を行う
        predicted_gf = int(round(model_gf.predict(example_match_df)[0], 0))
        predicted_ga = int(round(model_ga.predict(example_match_df)[0], 0))
        predictions = (predicted_gf, predicted_ga)
        
        selected_teams = (user_team, opponent_team)

        # ChatGPTによる質問応答
        if question: 
            team_info = f"ユーザーチーム: {user_team}, 対戦相手: {opponent_team}, 会場: {venue}"
            chat_response = ask_chatgpt(question, team_info, predicted_gf, predicted_ga)

    return render_template('index.html', teams=teams, venues=venues, predictions=predictions, selected_teams=selected_teams, chat_response=chat_response)

if __name__ == '__main__':
    app.run(debug=True)