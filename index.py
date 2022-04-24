import os
import logging
import http.client
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, render_template
import math
import random
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from pandas_datareader import data as pdr
import numpy as np
import json
import ast
import statistics
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from google.cloud import storage
import gcsfs

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'000-300418-51d36c16288b.json'

bucket_name = 'bucket'

storage_client = storage.Client()

app = Flask(__name__)

def doRender(tname, values={}):
	if not os.path.isfile( os.path.join(os.getcwd(), 'templates/'+tname) ): #No such file
		return render_template('form.htm')
	return render_template(tname, **values) 


@app.route('/calculate', methods=['POST'])
def RandomHandler():
        import http.client
        if request.method == 'POST':
                minhistory = request.form.get('minhistory')
                minhistory = int(minhistory)
                minhistory2 = str(minhistory)
                shots = request.form.get('shots')
                L = request.form.get("signal")
                Re = request.form.get("resources")
                sub = request.form.get("submit")
                yf.pdr_override()
                today = date.today()
                decadeAgo = today - timedelta(days=3652)
                data = pdr.get_data_yahoo('CSCO', start=decadeAgo, end=today)
                df_dates = data.index
                data['Buy']=0
                data['Sell']=0
                for i in range(len(data)):
                    realbody=math.fabs(data.Open[i]-data.Close[i])
                    bodyprojection=0.1*math.fabs(data.Close[i]-data.Open[i])
                	
                    if data.High[i] >= data.Close[i] and data.High[i]-bodyprojection <= data.Close[i] and data.Close[i] > data.Open[i] and data.Open[i] > data.Low[i] and data.Open[i]-data.Low[i] > realbody:
                        data.at[data.index[i], 'Buy'] = 1
                    if data.High[i] > data.Close[i] and data.High[i]-data.Close[i] > realbody and data.Close[i] > data.Open[i] and data.Open[i] >= data.Low[i] and data.Open[i] <= data.Low[i]+bodyprojection:
                        data.at[data.index[i], 'Buy'] = 1
                    if data.High[i] >= data.Open[i] and data.High[i]-bodyprojection <= data.Open[i] and data.Open[i] > data.Close[i] and data.Close[i] > data.Low[i] and data.Close[i]-data.Low[i] > realbody:
                        data.at[data.index[i], 'Sell'] = 1
                    if data.High[i] > data.Open[i] and data.High[i]-data.Open[i] > realbody and data.Open[i] > data.Close[i] and data.Close[i] >= data.Low[i] and data.Close[i] <= data.Low[i]+bodyprojection:
                        data.at[data.index[i], 'Sell'] = 1
                data_close = [entry[3] for entry in data.pct_change(1).values.tolist()]
                data_buy = [entry[6] for entry in data.values.tolist()]
                data_sell = [entry[7] for entry in data.values.tolist()]
                dates = [entry for entry in df_dates.tolist()]
                var95 = []
                var99 = []
                dt = []
                if L == "Buy":
                    for i in range(minhistory, len(data)):
                        if data_buy[i] == 1:
                            t = time.time()
                            mean = statistics.mean(data_close[i-minhistory:i])
                            std = statistics.stdev(data_close[i-minhistory:i])
                            dated = dates[i]
                            c = http.client.HTTPSConnection("80ugty989ujg.execute-api.us-east-1.amazonaws.com")
                            params = '{ "shots": "'+shots+'", "key1": "'+str(mean)+'", "key2": "'+str(std)+'"}'
                            json_input = json.dumps(params)
                            c.request("POST", "/default/function_one", params)
                            response = c.getresponse()
                            data = response.read().decode('utf-8')
                            print(data)
                            t1 = time.time() - t
                            t2 = 0.0000000021 * 1000
                            cost = t1 * t2
                            data = ast.literal_eval(data)
                            var95.append(data[0])
                            var99.append(data[1])
                            dt.append(dated)
                            mn1 = pd.DataFrame(var95).mean()
                            mn2 = pd.DataFrame(var99).mean()
                            dframe = pd.DataFrame(dt, columns = ['Date'])
                            dframe['Var95'] = pd.DataFrame(var95)
                            dframe['Var99'] = pd.DataFrame(var99)
                            dfObj = pd.DataFrame(columns=['M', 'H', 'L', 'R', 'Service', 'Time', 'Costs'])
                            dfObj = dfObj.append({'M': str(minhistory), 'H': str(shots), 'L': L, 'R': Re, 'Service': sub, 'Time': t1, 'Costs': cost}, ignore_index=True)
                            print(dfObj)
                            d = zip(dt, var95, var99)
                            upload(dframe, dfObj)
                    mea1 = []
                    mea2 = []
                    indexer = []
                    vs95 = dframe['Var95']
                    vs99 = dframe['Var99']
                    me1 = vs95.mean()
                    me2 = vs99.mean()
                    i = 0
                    for i in range(i, len(vs95)): 
                        mea1.append(me1)
                    i = 0
                    for i in range(i, len(vs99)): 
                        mea2.append(me2) 
                    i = 0
                    for i in range(i, len(vs99)): 
                        indexer.append(i)
                    i = 0
                    for i in range(i, len(vs95)): 
                        vs95[i] = float(vs95[i])
                    i = 0
                    for i in range(i, len(vs99)): 
                        vs99[i] = float(vs99[i])    
                    vs95 = vs95.to_numpy().tolist()
                    vs99 = vs99.to_numpy().tolist()                          
                    print(vs95, vs99, mea1, mea2, indexer)
                    v95 = np.array(var95)
                    v99 = np.array(var99)
                    cc = chart(v95, v99)          
                    return doRender( 'form.htm',
                        {'note': d, 'mn1': mn1[0], 'mn2': mn2[0], 'vs95': vs95, 'vs99': vs99, 'mea1':mea1, 'mea2': mea2, 'indexer': indexer, 'time': t1, 'costs': cost} )        
                            
                elif L == "Sell":
                    for i in range(minhistory, len(data)):
                        if data_sell[i] == 1:
                            t = time.time()
                            mean = statistics.mean(data_close[i-minhistory:i])
                            std = statistics.stdev(data_close[i-minhistory:i])
                            dated = dates[i]
                            c = http.client.HTTPSConnection("80ugty989ujg.execute-api.us-east-1.amazonaws.com")
                            params = '{ "shots": "'+shots+'", "key1": "'+str(mean)+'", "key2": "'+str(std)+'"}'
                            json_input = json.dumps(params)
                            c.request("POST", "/default/function_one", params)
                            response = c.getresponse()
                            data = response.read().decode('utf-8')
                            print(data)
                            t1 = time.time() - t
                            t2 = 0.0000000021 * 1000
                            cost = t1 * t2
                            data = ast.literal_eval(data)
                            var95.append(data[0])
                            var99.append(data[1])
                            dt.append(dated)
                            mn1 = pd.DataFrame(var95).mean()
                            mn2 = pd.DataFrame(var99).mean()
                            dframe = pd.DataFrame(dt, columns = ['Date'])
                            dframe['Var95'] = pd.DataFrame(var95)
                            dframe['Var99'] = pd.DataFrame(var99)
                            dfObj = pd.DataFrame(columns=['M', 'H', 'L', 'R', 'Service', 'Time', 'Costs'])
                            dfObj = dfObj.append({'M': str(minhistory), 'H': str(shots), 'L': L, 'R': Re, 'Service': sub, 'Time': t1, 'Costs': cost}, ignore_index=True)
                            print(dfObj)
                            d = zip(dt, var95, var99)
                            upload(dframe, dfObj)
                    mea1 = []
                    mea2 = []
                    indexer = []
                    vs95 = dframe['Var95']
                    vs99 = dframe['Var99']
                    me1 = vs95.mean()
                    me2 = vs99.mean()
                    i = 0
                    for i in range(i, len(vs95)): 
                        mea1.append(me1)
                    i = 0
                    for i in range(i, len(vs99)): 
                        mea2.append(me2) 
                    i = 0
                    for i in range(i, len(vs99)): 
                        indexer.append(i)
                    i = 0
                    for i in range(i, len(vs95)): 
                        vs95[i] = float(vs95[i])
                    i = 0
                    for i in range(i, len(vs99)): 
                        vs99[i] = float(vs99[i])    
                    vs95 = vs95.to_numpy().tolist()
                    vs99 = vs99.to_numpy().tolist()                          
                    print(vs95, vs99, mea1, mea2, indexer)
                    v95 = np.array(var95)
                    v99 = np.array(var99)        
                    return doRender( 'form.htm',
                        {'note': d, 'mn1': mn1[0], 'mn2': mn2[0], 'vs95': vs95, 'vs99': vs99, 'mea1':mea1, 'mea2': mea2, 'indexer': indexer, 'time': t1, 'costs': cost} )
                                             
    
    
def upload(ex, ex2):

    bucket = storage_client.get_bucket('bucket')
    blob = bucket.blob("newaudit1.csv")
    blob2 = bucket.blob("newaudit2.csv")
    blob.upload_from_string(ex.to_csv(), 'text/csv')
    blob2.upload_from_string(ex2.to_csv(), 'text/csv')      
    
    return 1
    

@app.route('/history')
def storage(): 

    fs = gcsfs.GCSFileSystem(project='ooo-300418')
    with fs.open('bucket/newaudit1.csv') as f:
        dff = pd.read_csv(f)
        dff1 = dff.to_html()
    with fs.open('bucket/newaudit2.csv') as f1:
        dff2 = pd.read_csv(f1)
        dff3 = dff2.to_html()
    v95 = dff.iloc[:, 2]
    v99 = dff.iloc[:, 3]    
    achart = chart(v95, v99)
 
    return doRender('history.htm',
                        {'note2': dff1, 'note3': dff3, 'achart': achart } )


# catch all other page requests - doRender checks if a page is available (shows it) or not (index)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def mainPage(path):
	return doRender(path)

@app.errorhandler(500)
# A small bit of error handling
def server_error(e):
    logging.exception('ERROR!')
    return """
    An  error occurred: <pre>{}</pre>
    """.format(e), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)  
