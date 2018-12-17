from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from constants import *
from preprocess_data import dump_variable, load_variable


def main():
    classifiers = {
        'logistic_regression': LogisticRegression(random_state=0),
        'knn': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        'svc': SVC(kernel='linear', random_state=0),
        'naive_bayes': GaussianNB(),
        'decision_tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
        'random_forrest': RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0),
    }

    X = load_variable(X_FILE)
    y = load_variable(Y_FILE)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    for classifier_name, classifier in classifiers.items():
        print(f'training {classifier_name}...')
        classifier.fit(X_train, y_train)

        dump_variable(classifier, f'classifiers/{classifier_name}.pickle')

    print()

    metrics = []

    for classifier_name, classifier in classifiers.items():
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        tp, tn, fp, fn = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]

        print(classifier_name)
        print(f'\ttrue positives: {tp}')
        print(f'\ttrue negatives: {tn}')
        print(f'\tfalse positives: {fp}')
        print(f'\tfalse negatives: {fn}')

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print(f'\tprecision: {precision}')
        print(f'\trecall: {recall}')

        f1_score = 2 * precision * recall / (precision + recall)

        print(f'\tf1 score: {f1_score}')

        metrics.append((f1_score, classifier_name))

    metrics = sorted(metrics, key=lambda x: x[0], reverse=True)
    print()

    print('metrics:')
    for f1_score, classifier_name in metrics:
        print(f'{classifier_name}: {f1_score}')


if __name__ == '__main__':
    main()
