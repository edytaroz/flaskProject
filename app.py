from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
import openX

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template('index.html')


@app.route('/choice', methods=['GET', 'POST'])
def choice():
    data = 0
    if request.method == 'POST':
        title = request.form['sel']
        X_train, X_test, y_train, y_test = openX.preprocess()
        if title == "o1":
            data = openX.heuristic(y_test)
        elif title == "o2":
            data = openX.knnclassifier(X_train, X_test, y_train, y_test)
            data2 = openX.dtclassifier(X_train, X_test, y_train, y_test)
            return "K nearest neighbors accuracy: " + str(accuracy_score(y_test,data)) + "\n" + "Decision tree accuracy: " + str(accuracy_score(y_test,data2))
        elif title == "o3":
            data = openX.nnclassifier(X_train, X_test, y_train, y_test)
        return "Accuracy: " + str(accuracy_score(y_test,data))


if __name__ == '__main__':
    app.run()
