import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib


app = Flask(__name__)
nba_data = pd.read_csv('nba_2022-23_all_stats_with_salary.csv')
nba_data['GSPERCENT'] = nba_data['GS'] / nba_data['GP']

pg_data = nba_data[nba_data['Position'] == 'PG']
sg_data = nba_data[nba_data['Position'] == 'SG']
sf_data = nba_data[nba_data['Position'] == 'SF']
pf_data = nba_data[nba_data['Position'] == 'PF']
c_data = nba_data[nba_data['Position'] == 'C']

pg_feat = pg_data[['Age', 'PTS', 'AST', 'FG', 'Total Minutes', 'GSPERCENT']]
pg_salary = pg_data['Salary']

sg_feat = sg_data[['Age', 'PTS', 'AST', 'FG', 'Total Minutes', 'GSPERCENT']]
sg_salary = sg_data['Salary']

sf_feat = sf_data[['Age', 'PTS', 'AST', 'TRB', 'Total Minutes', 'GSPERCENT']]
sf_salary = sf_data['Salary']

pf_feat = pf_data[['Age', 'PTS', 'TRB', 'BLK', 'Total Minutes', 'GSPERCENT']]
pf_salary = pf_data['Salary']

c_feat = c_data[['Age', 'PTS', 'TRB', 'BLK', 'Total Minutes', 'GSPERCENT']]
c_salary = c_data['Salary']

@app.route('/player/PG', methods=['POST'])
def handle_pg():
    # PG Model
    player_name = request.json.get('playerName')
    print(player_name)
    return jsonify({"playerName": player_name})
    pg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pg_feat_train, pg_feat_test, pg_predict_train, pg_predict_test = train_test_split(pg_feat, pg_salary, test_size=0.2, random_state=42)
    pg_model.fit(pg_feat_train, pg_predict_train)
    predicted_pg_salaries = pg_model.predict(pg_feat_test)

@app.route('/player/SG', methods=['POST'])
def handle_sg():
    # SG Model
    sg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sg_feat_train, sg_feat_test, sg_predict_train, sg_predict_test = train_test_split(sg_feat, sg_salary, test_size=0.2, random_state=42)
    sg_model.fit(sg_feat_train, sg_predict_train)
    predicted_sg_salaries = sg_model.predict(sg_feat_test)
    data = predicted_sg_salaries.get_json()
    return jsonify(data)

@app.route('/player/SF', methods=['POST'])
def handle_sf():
    sf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    sf_feat_train, sf_feat_test, sf_predict_train, sf_predict_test = train_test_split(sf_feat, sf_salary, test_size=0.2, random_state=42)
    sf_model.fit(sf_feat_train, sf_predict_train)
    predicted_sf_salaries = sf_model.predict(sf_feat_test)
    data = predicted_sf_salaries.get_json()
    return jsonify(data)

@app.route('/player/PF', methods=['POST'])
def handle_pf():
    pf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pf_feat_train, pf_feat_test, pf_predict_train, pf_predict_test = train_test_split(pf_feat, pf_salary, test_size=0.2, random_state=42)
    pf_model.fit(pf_feat_train, pf_predict_train)
    predicted_pf_salaries = pf_model.predict(pf_feat_test)
    data = predicted_pf_salaries.get_json()
    return jsonify(data)

@app.route('/player/C', methods=['POST'])
def handle_c():
    c_model = RandomForestRegressor(n_estimators=100, random_state=42)
    c_feat_train, c_feat_test, c_predict_train, c_predict_test = train_test_split(c_feat, c_salary, test_size=0.2, random_state=42)
    c_model.fit(c_feat_train, c_predict_train)
    predicted_c_salaries = c_model.predict(c_feat_test)
    data = predicted_c_salaries.get_json()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)