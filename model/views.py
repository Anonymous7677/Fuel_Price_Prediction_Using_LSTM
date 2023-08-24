from django.shortcuts import render
from django.http import HttpResponse
from pathlib import Path
from build.main import *

BASE_DIR = Path(__file__).resolve().parent.parent


# Create your views here.
def say_hello(request):
    _ = get_latest_dataset()
    _ = update_unified_dataset()
    
    pred_95, pred_diesel = predict_ron95_diesel()
    pred_97 = predict_ron97()
    date = save_predictions(pred_95, pred_97, pred_diesel)

    Sequence = {
        'date_1': date[0],
        'date_2': date[1],
        'date_3': date[2],
        'r95_1': pred_95[0],
        'r95_2': pred_95[1],
        'r95_3': pred_95[2],
        'r97_1': pred_97[0],
        'r97_2': pred_97[1],
        'r97_3': pred_97[2],
        'diesel_1': pred_diesel[0],
        'diesel_2': pred_diesel[1],
        'diesel_3': pred_diesel[2]

    }
    

    return render(request, 'home.html', Sequence)


def load_history(request):

    df = pd.read_csv(history)
    file = str(BASE_DIR) + "/templates/history.html"

    html_string = '''
    <!DOCTYPE html>
    <html>

    <head>
        <title>History</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
        {{% load static %}}
        <link rel="stylesheet" href="{{% static '/style.css'%}}">
    </head>

    <body>
        <div class="nav">
            <p class="nav-title">
            Malaysia Fuel Price Prediction Using Artificial Neural Network
            </p>
        </div>

        <br />
        <br />
        <br />
        <br />

        <p class="table-title" style="text-align: left; margin-left: 10%; margin-top: 5%; font-size: 20px; font-family: consolas; color: black;">
            Prediction History
        </p>

        {table}

    </body>
    
    <footer>
        <nav class="footer-content">
            <p style="font-size:20px;"><a href="/model/home/"> Home</a> | <a href="/model/history/">History</a></p>
        </nav>
    </footer>
    </html>
    '''
    with open(file, 'w') as f_output:
        f_output.write(html_string.format(table=df.to_html()))

    return render(request, 'history.html')
