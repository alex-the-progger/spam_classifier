from preprocess_data import load_variable, vectorize_sentence

classifier = load_variable('classifiers/svc.pickle')


while True:
    sentence = input('Enter a sentence (or blank line to exit):\n')
    if not sentence:
        break

    vector = vectorize_sentence(sentence)
    result = classifier.predict([vector])[0]

    if result:
        print('This is spam')
    else:
        print('This is not spam')

    print()
