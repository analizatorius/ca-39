from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pickle

#Create Flask instance
app = Flask(__name__)
#create a secret key
app.config["SECRET_KEY"] = "much_secret"

#create form for iris
class IrisData(FlaskForm):
    sepal_length = StringField("Sepal length", validators=[DataRequired()])
    sepal_width = StringField("Sepal width", validators=[DataRequired()])
    petal_length = StringField("Petal length", validators=[DataRequired()])
    petal_width = StringField("Petal width", validators=[DataRequired()])
    submit = SubmitField("submit")

#Create route decorator for index
@app.route('/')
def index():

    return render_template("index.html")


# Machine learning assignemnt
@app.route("/iris", methods=["GET"])
def iris():
    form = IrisData()
    return render_template("iris.html", form=form)


@app.route("/iris", methods=["POST"])
def iris_post():
    form = IrisData()

    sepal_length = form.sepal_length.data
    sepal_width = form.sepal_width.data
    petal_length = form.petal_length.data
    petal_width = form.petal_width.data

    prediction = iris_prediction([sepal_length, sepal_width, petal_length, petal_width])
    prediction = str(prediction[0])

    image = "https://upload.wikimedia.org/wikipedia/commons/f/f8/Iris_virginica_2.jpg"

    return render_template("iris_get.html", form=form, prediction=prediction, image=image)


def iris_prediction(iris_data: list):
    pickled_model = open('iris_predictor.pickle', 'rb')
    model = pickle.load(pickled_model)
    prediction = model.predict([iris_data])

    return prediction


if __name__ == '__main__':
    app.run(debug=True)