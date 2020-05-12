import matplotlib.pyplot as plt

import data_helpers
import Train


def POS_Tagger_hyperparameter_optimization(cv, crf_iteration, search_iteration, L1_coefficient, L2_coefficient):
    X_train, y_train, X_test, y_test = data_helpers.load_POS_data(None, False)
    coefficients = Train.hyperparameter_optimization(X_train,y_train,X_test,y_test, crf_iteration, search_iteration, cv, L1_coefficient, L2_coefficient, "POS_Tagger")
    return coefficients


def Chunker_hyperparameter_optimization(cv, crf_iteration, search_iteration, L1_coefficient, L2_coefficient):
    X_train, y_train, X_test, y_test = data_helpers.load_Chunk_data(None, False)
    coefficients = Train.hyperparameter_optimization(X_train,y_train,X_test,y_test, crf_iteration, search_iteration, cv, L1_coefficient, L2_coefficient, "Chunker")
    return coefficients

def POS_Tagger(crf_iteration, L1_coefficient, L2_coefficient):
    X_train, y_train, X_test, y_test = data_helpers.load_POS_data(None, True)
    labels, crf = Train.train(X_train, y_train, crf_iteration, L1_coefficient, L2_coefficient)
    y_pred = Train.Predict(X_test, crf)
    Train.Classification_report_detail(y_test, y_pred)
    print("POS_Tagger accuracy:", Train.Classification_report_accuracy(y_test, y_pred))
    print("POS_Tagger weighted f-score:", Train.Classification_report_fscore_weighted(y_test, y_pred))
    print("POS_Tagger macro f-score:", Train.Classification_report_fscore_macro(y_test, y_pred))
    print("POS_Tagger micro f-score:", Train.Classification_report_fscore_micro(y_test, y_pred))

def Chunker(crf_iteration, L1_coefficient, L2_coefficient):
    X_train, y_train, X_test, y_test = data_helpers.load_Chunk_data(None, True)
    labels, crf = Train.train(X_train, y_train, crf_iteration, L1_coefficient, L2_coefficient)
    y_pred = Train.Predict(X_test, crf)

    Train.Classification_report_detail(y_test, y_pred)
    print("Chunker accuracy:", Train.Classification_report_accuracy(y_test, y_pred))
    print("Chunker weighted f-score:", Train.Classification_report_fscore_weighted(y_test, y_pred))
    print("Chunker macro f-score:", Train.Classification_report_fscore_macro(y_test, y_pred))
    print("Chunker micro f-score:", Train.Classification_report_fscore_micro(y_test, y_pred))


def Chunker_without_POS(crf_iteration, L1_coefficient, L2_coefficient):
    X_train, y_train, X_test, y_test = data_helpers.load_Chunk_data_without_POS(True)
    labels, crf = Train.train(X_train, y_train, crf_iteration, L1_coefficient, L2_coefficient)
    y_pred = Train.Predict(X_test, crf)

    Train.Classification_report_detail(y_test, y_pred)
    print("Chunker accuracy:", Train.Classification_report_accuracy(y_test, y_pred))
    print("Chunker weighted f-score:", Train.Classification_report_fscore_weighted(y_test, y_pred))
    print("Chunker macro f-score:", Train.Classification_report_fscore_macro(y_test, y_pred))
    print("Chunker micro f-score:", Train.Classification_report_fscore_micro(y_test, y_pred))


def Trained_POS2Chunker(crf_iteration, L1_coefficient_p, L2_coefficient_p, L1_coefficient_c, L2_coefficient_c):
    X_train, y_train, X_test, y_test = data_helpers.load_POS_data(None, True)
    _, crf = Train.train(X_train, y_train, crf_iteration, L1_coefficient_p, L2_coefficient_p)
    y_pred_pos = Train.Predict(X_train, crf)

    X_train, y_train, X_test, y_test = data_helpers.load_Chunk_data(y_pred_pos, False)
    labels, crf = Train.train(X_train, y_train, crf_iteration, L1_coefficient_c, L2_coefficient_c)

    y_pred = Train.Predict(X_test, crf)

    Train.Classification_report_detail(y_test, y_pred)
    print("Chunker accuracy:", Train.Classification_report_accuracy(y_test, y_pred))
    print("Chunker weighted f-score:", Train.Classification_report_fscore_weighted(y_test, y_pred))
    print("Chunker macro f-score:", Train.Classification_report_fscore_macro(y_test, y_pred))
    print("Chunker micro f-score:", Train.Classification_report_fscore_micro(y_test, y_pred))




def Pred_POS2Chunker():
    pass

def Iterative_Training(iteration, crf_iteration, L1_coefficient_p, L2_coefficient_p, L1_coefficient_c, L2_coefficient_c):
    X_train_pos, y_train_pos, X_test_pos, y_test_pos = data_helpers.load_POS_data(None, True)

    pos_fscores = []
    chunk_fscores = []
    for i in range(iteration):
        _, crf_pos = Train.train(X_train_pos, y_train_pos, crf_iteration, L1_coefficient_p, L2_coefficient_p)
        y_pred_pos = Train.Predict(X_train_pos, crf_pos)
        y_pred_pos_f = Train.Predict(X_test_pos, crf_pos)

        X_train_chunk, y_train_chunk, X_test_chunk, y_test_chunk = data_helpers.load_Chunk_data(y_pred_pos, False)
        _, crf_chunk = Train.train(X_train_chunk, y_train_chunk, crf_iteration, L1_coefficient_c, L2_coefficient_c)

        y_pred_chunk = Train.Predict(X_train_chunk, crf_chunk)
        y_pred_chunk_f = Train.Predict(X_test_chunk, crf_chunk)
        X_train_pos, y_train_pos, X_test_pos, y_test_pos = data_helpers.load_POS_data_with_chunk(y_pred_chunk, False)
        pos_fscores.append(round(Train.Classification_report_fscore_weighted(y_test_pos, y_pred_pos_f),5))
        chunk_fscores.append(round(Train.Classification_report_fscore_weighted(y_test_chunk, y_pred_chunk_f),5))
        print("Iteration,", i+1, "POS_Tagger weighted f-score:", Train.Classification_report_fscore_weighted(y_test_pos, y_pred_pos_f),
              ", Chunker weighted f-score:", Train.Classification_report_fscore_weighted(y_test_chunk, y_pred_chunk_f))

    print("Classification report after", iteration, "iteration")
    Train.Classification_report_detail(y_test_pos, y_pred_pos_f)
    print("POS_Tagger accuracy:", Train.Classification_report_accuracy(y_test_pos, y_pred_pos_f))
    print("POS_Tagger weighted f-score:", Train.Classification_report_fscore_weighted(y_test_pos, y_pred_pos_f))
    print("POS_Tagger macro f-score:", Train.Classification_report_fscore_macro(y_test_pos, y_pred_pos_f))
    print("POS_Tagger micro f-score:", Train.Classification_report_fscore_micro(y_test_pos, y_pred_pos_f))
    print("________________________________________________________________________________")
    Train.Classification_report_detail(y_test_chunk, y_pred_chunk_f)
    print("Chunker accuracy:", Train.Classification_report_accuracy(y_test_chunk, y_pred_chunk_f))
    print("Chunker weighted f-score:", Train.Classification_report_fscore_weighted(y_test_chunk, y_pred_chunk_f))
    print("Chunker macro f-score:", Train.Classification_report_fscore_macro(y_test_chunk, y_pred_chunk_f))
    print("Chunker micro f-score:", Train.Classification_report_fscore_micro(y_test_chunk, y_pred_chunk_f))

    plt.plot(pos_fscores)
    plt.plot(chunk_fscores)
    plt.title("CRF iteration: " + str(crf_iteration))
    plt.ylabel('weighted f-score')
    plt.xlabel('iteration')
    plt.legend(['POS test', 'Chunk test'], loc='upper left')
    plt.figtext(.2, .2, " POS Max: " + str(max(pos_fscores) * 100) + " in iteration: " + str(pos_fscores.index(max(pos_fscores)) + 1) +" Chunk Max: " + str(max(chunk_fscores) * 100) + " in iteration: " + str(chunk_fscores.index(max(chunk_fscores)) + 1))
    plt.show()






if __name__ == "__main__":
    # model_type = 'fixed'
    model_type = 'tuned'

    crf_iteration = 100

    if model_type == 'fixed':
        L1_coefficient = 0.1
        L2_coefficient = 0.1
        POS_Tagger(crf_iteration, L1_coefficient, L2_coefficient)
        # Chunker_without_POS(crf_iteration, L1_coefficient, L2_coefficient)
        # Chunker(crf_iteration, L1_coefficient, L2_coefficient)
        # Train.hyperparameter_optimization(X_train, y_train, X_test, y_test)
        # Trained_POS2Chunker(crf_iteration, L1_coefficient, L2_coefficient, L1_coefficient, L2_coefficient)
        # Iterative_Training(10, crf_iteration, L1_coefficient, L2_coefficient, L1_coefficient, L2_coefficient)

    if model_type == 'tuned':
        cv = 3
        search_iteration = 50
        L1_coefficient = 0.5
        L2_coefficient = 0.05


        POS_Tagger_coefficients = POS_Tagger_hyperparameter_optimization(cv, crf_iteration, search_iteration, L1_coefficient, L2_coefficient)
        Chunker_coefficients = Chunker_hyperparameter_optimization(cv, crf_iteration, search_iteration, L1_coefficient, L2_coefficient)

        POS_Tagger_L1_coefficient = POS_Tagger_coefficients['c1']
        POS_Tagger_L2_coefficient = POS_Tagger_coefficients['c2']
        Chunker_L1_coefficient = Chunker_coefficients['c1']
        Chunker_L2_coefficient = Chunker_coefficients['c2']

        # POS_Tagger(crf_iteration, POS_Tagger_L1_coefficient, POS_Tagger_L2_coefficient)
        #Chunker(crf_iteration, Chunker_L1_coefficient, Chunker_L2_coefficient)
        #Trained_POS2Chunker(crf_iteration, POS_Tagger_L1_coefficient, POS_Tagger_L2_coefficient, Chunker_L1_coefficient, Chunker_L2_coefficient)
        Iterative_Training(10, crf_iteration, POS_Tagger_L1_coefficient, POS_Tagger_L2_coefficient, Chunker_L1_coefficient, Chunker_L2_coefficient)



