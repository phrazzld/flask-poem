from flask import Flask, render_template, session, request, flash

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
app.config.from_pyfile('config.py')

import label_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # run prediction, save to session
        image_url = request.form['img-url']
        session['prediction'], session['image_path'] = label_image.label(image_url)

    prediction = session.get('prediction', '')
    image_path = session.get('image_path', '')
    return render_template('index.html',
                           prediction=prediction,
                           image_path=image_path)

if __name__ == '__main__':
    app.run()
