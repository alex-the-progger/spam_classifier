from constants import *
from preprocess_data import load_variable

classifier = load_variable('classifiers/svc.pickle')


while True:
    sentence = input('Enter a sentence (or blank line to exit):\n')
    if not sentence:
        break

    text_encoder = load_variable(TEXT_ENCODER_FILE)

    vector = text_encoder.transform([sentence])
    result = classifier.predict(vector)[0]

    if result:
        print('This is spam')
    else:
        print('This is not spam')

    print()
