def main():
    import sys
    sys.path.append('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/data_noise_variance')
    import os  
    import numpy as np
    import argparse
    import keras.backend as K
    import gc
    from neural_networks import MVENetwork
    from klepto.archives import dir_archive
    from utils import rmse, average_loglikelihood, maxdiagonal, load_data
    from sklearn.model_selection import KFold

    os.chdir('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2022/data_noise_variance')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,
                        help="name of dataset")
    parser.add_argument('-hid', '--n_hidden', type=int,
                        nargs='+', required=True, help="Number hidden units per layer")
    parser.add_argument('-ne', '--number_of_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('-nfo', '--number_of_folds_out', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-nfi', '--number_of_folds_in', type=int, default=10,
                        help='Number of training epochs')
    args = parser.parse_args()

    dataset = args.dataset
    results_UCI = dir_archive(f'./UCI_experiment/{dataset}', serialized=True)
    results_UCI.load()
    X, Y = load_data(dataset)
    n_epochs = args.number_of_epochs
    n_hidden = np.array(args.n_hidden)
    number_of_outer_folds = args.number_of_folds_out
    number_of_inner_folds = args.number_of_folds_in
    reg_factors = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results_UCI = {'LL_nowarmup_same': np.zeros(number_of_outer_folds),
                            'LL_nowarmup_separate': np.zeros(number_of_outer_folds),
                            'LL_warmup_same': np.zeros(number_of_outer_folds),
                            'LL_warmup_separate': np.zeros(number_of_outer_folds),
                            'LL_warmup_fixedmean_same': np.zeros(number_of_outer_folds),
                            'LL_warmup_fixedmean_separate': np.zeros(number_of_outer_folds),
                            'rmse_nowarmup_same': np.zeros(number_of_outer_folds),
                            'rmse_nowarmup_separate': np.zeros(number_of_outer_folds),
                            'rmse_warmup_same': np.zeros(number_of_outer_folds),
                            'rmse_warmup_separate': np.zeros(number_of_outer_folds),
                            'rmse_warmup_fixedmean_same': np.zeros(number_of_outer_folds),
                            'rmse_warmup_fixedmean_separate': np.zeros(number_of_outer_folds),
                            'regconstant_nowarmup_same': np.zeros((number_of_outer_folds, 2)),
                            'regconstant_nowarmup_separate': np.zeros((number_of_outer_folds, 2)),
                            'regconstant_warmup_same': np.zeros((number_of_outer_folds, 2)),
                            'regconstant_warmup_separate': np.zeros((number_of_outer_folds, 2)),
                            'regconstant_warmup_fixedmean_same': np.zeros((number_of_outer_folds, 2)),
                            'regconstant_warmup_fixedmean_separate': np.zeros((number_of_outer_folds, 2)),
                            'epochs':n_epochs,
                            'architecture':n_hidden,
                            }
    for i, (training_indices, test_indices) in enumerate(KFold(number_of_outer_folds, shuffle=True, random_state=1).split(X)):
        print(f'{i + 1}  of {number_of_outer_folds}')
        x, x_val = X[training_indices], X[test_indices]
        y, y_val = Y[training_indices], Y[test_indices]
        # 10-fold cross validation to find the optimal regularization
        cvalidation_nowarmup = {(a, b): 0 for a in reg_factors for b in reg_factors}
        cvalidation_warmup = {(a, b): 0 for a in reg_factors for b in reg_factors}
        cvalidation_warmup_fixedmean = {(a, b): 0 for a in reg_factors for b in reg_factors}
        for training_indices_2, test_indices_2 in KFold(number_of_inner_folds, shuffle=True, random_state=5).split(x):
            x_train, x_test = x[training_indices_2], x[test_indices_2]
            y_train, y_test = y[training_indices_2], y[test_indices_2]
            for a in reg_factors:
                for b in reg_factors:
                    model_nowarmup = MVENetwork(X=x_train, Y=y_train, n_hidden_mean=n_hidden,
                                                n_hidden_var=n_hidden, n_epochs=n_epochs, reg_mean=a, reg_var=b,
                                                batch_size=None, verbose=0, warmup=0)
                    model_warmup = MVENetwork(X=x_train, Y=y_train, n_hidden_mean=n_hidden,
                                              n_hidden_var=n_hidden, n_epochs=n_epochs, reg_mean=a, reg_var=b,
                                              batch_size=None, verbose=0, warmup=1, fixed_mean=0)
                    model_warmup_fixedmean = MVENetwork(X=x_train, Y=y_train, n_hidden_mean=n_hidden,
                                                        n_hidden_var=n_hidden, n_epochs=n_epochs, reg_mean=a,
                                                        reg_var=b,
                                                        batch_size=None, verbose=0, warmup=1, fixed_mean=1)
                    cvalidation_nowarmup[(a, b)] += average_loglikelihood(y_test, model_nowarmup.f(x_test),
                                                                          model_nowarmup.sigma(x_test))
                    cvalidation_warmup[(a, b)] += average_loglikelihood(y_test, model_warmup.f(x_test),
                                                                        model_warmup.sigma(x_test))
                    cvalidation_warmup_fixedmean[(a, b)] += average_loglikelihood(y_test,
                                                                                  model_warmup_fixedmean.f(x_test),
                                                                                  model_warmup_fixedmean.sigma(x_test))
                    gc.collect()
                    K.clear_session()
        # Picking the optimal regularization constants
        best_regularization_nowarmup_separate = list(max(cvalidation_nowarmup, key=cvalidation_nowarmup.get))
        best_regularization_nowarmup_same = maxdiagonal(cvalidation_nowarmup)
        best_regularization_warmup_separate = list(max(cvalidation_warmup, key=cvalidation_warmup.get))
        best_regularization_warmup_same = maxdiagonal(cvalidation_warmup)
        best_regularization_warmup_fixedmean_separate = list(max(cvalidation_warmup_fixedmean, key=cvalidation_warmup_fixedmean.get))
        best_regularization_warmup_fixedmean_same = maxdiagonal(cvalidation_warmup_fixedmean)

        # Training models using the best found regularization constants
        model_nowarmup_separate = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                             n_hidden_var=n_hidden, n_epochs=n_epochs,
                                             reg_mean=best_regularization_nowarmup_separate[0],
                                             reg_var=best_regularization_nowarmup_separate[1],
                                             batch_size=None, verbose=0, warmup=0)

        # If the optimal regularization for the mean and variance are equal we take the same network
        if best_regularization_nowarmup_separate[0] == best_regularization_nowarmup_separate[1]:
            model_nowarmup_same = model_nowarmup_separate
        else:
            model_nowarmup_same = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                             n_hidden_var=n_hidden, n_epochs=n_epochs,
                                             reg_mean=best_regularization_nowarmup_same[0],
                                             reg_var=best_regularization_nowarmup_same[1],
                                             batch_size=None, verbose=0, warmup=0)
        # With warmup
        model_warmup_separate = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                           n_hidden_var=n_hidden, n_epochs=n_epochs,
                                           reg_mean=best_regularization_warmup_separate[0],
                                           reg_var=best_regularization_warmup_separate[1],
                                           batch_size=None, verbose=0, warmup=1, fixed_mean=0)
        if best_regularization_warmup_separate[0] == best_regularization_warmup_separate[1]:
            model_warmup_same = model_warmup_separate
        else:
            model_warmup_same = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                           n_hidden_var=n_hidden, n_epochs=n_epochs,
                                           reg_mean=best_regularization_warmup_same[0],
                                           reg_var=best_regularization_warmup_same[1],
                                           batch_size=None, verbose=0, warmup=1, fixed_mean=0)

        # With warmup and with fixed mean
        model_warmup_fixedmean_separate = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                                     n_hidden_var=n_hidden, n_epochs=n_epochs,
                                                     reg_mean=best_regularization_warmup_fixedmean_separate[0],
                                                     reg_var=best_regularization_warmup_fixedmean_separate[1],
                                                     batch_size=None, verbose=0, warmup=1, fixed_mean=1)
        if best_regularization_warmup_fixedmean_separate[0] == best_regularization_warmup_fixedmean_separate[1]:
            model_warmup_fixedmean_same = model_warmup_fixedmean_separate
        else:
            model_warmup_fixedmean_same = MVENetwork(X=x, Y=y, n_hidden_mean=n_hidden,
                                                     n_hidden_var=n_hidden, n_epochs=n_epochs,
                                                     reg_mean=best_regularization_warmup_fixedmean_same[0],
                                                     reg_var=best_regularization_warmup_fixedmean_same[1],
                                                     batch_size=None, verbose=0, warmup=1, fixed_mean=1)

        # Calculating all relevant metrics for the 6 models
        results_UCI['LL_nowarmup_same'][i] = average_loglikelihood(y_val,
                                                                            model_nowarmup_same.f(x_val),
                                                                            model_nowarmup_same.sigma(x_val))
        results_UCI['LL_nowarmup_separate'][i] = average_loglikelihood(y_val,
                                                                                model_nowarmup_separate.f(x_val),
                                                                                model_nowarmup_separate.sigma(x_val))
        results_UCI['LL_warmup_same'][i] = average_loglikelihood(y_val,
                                                                          model_warmup_same.f(x_val),
                                                                          model_warmup_same.sigma(x_val))
        results_UCI['LL_warmup_separate'][i] = average_loglikelihood(y_val,
                                                                              model_warmup_separate.f(x_val),
                                                                              model_warmup_separate.sigma(x_val))
        results_UCI['LL_warmup_fixedmean_same'][i] = average_loglikelihood(y_val,
                                                                                    model_warmup_fixedmean_same.f(x_val),
                                                                                    model_warmup_fixedmean_same.sigma(x_val))
        results_UCI['LL_warmup_fixedmean_separate'][i] = average_loglikelihood(y_val,
                                                                                        model_warmup_fixedmean_separate.f(x_val),
                                                                                        model_warmup_fixedmean_separate.sigma(x_val))

        results_UCI['rmse_nowarmup_same'][i] = rmse(y_val, model_nowarmup_same.f(x_val))
        results_UCI['rmse_nowarmup_separate'][i] = rmse(y_val, model_nowarmup_separate.f(x_val))
        results_UCI['rmse_warmup_same'][i] = rmse(y_val, model_warmup_same.f(x_val))
        results_UCI['rmse_warmup_separate'][i] = rmse(y_val, model_warmup_separate.f(x_val))
        results_UCI['rmse_warmup_fixedmean_same'][i] = rmse(y_val, model_warmup_fixedmean_same.f(x_val))
        results_UCI['rmse_warmup_fixedmean_separate'][i] = rmse(y_val, model_warmup_fixedmean_separate.f(x_val))

        results_UCI['regconstant_nowarmup_same'][i] = best_regularization_nowarmup_same
        results_UCI['regconstant_nowarmup_separate'][i] = best_regularization_nowarmup_separate
        results_UCI['regconstant_warmup_same'][i] = best_regularization_warmup_same
        results_UCI['regconstant_warmup_separate'][i] = best_regularization_warmup_separate
        results_UCI['regconstant_warmup_fixedmean_same'][i] = best_regularization_warmup_fixedmean_same
        results_UCI['regconstant_warmup_fixedmean_separate'][i] = best_regularization_warmup_fixedmean_separate

        gc.collect()
        K.clear_session()

    results_UCI['epochs'] = n_epochs
    results_UCI['architecture'] = n_hidden

    # Saving the directory containing all the results
    results = dir_archive(f'./UCI_experiment/{dataset}', results_UCI, serialized=True)
    results.dump()


if __name__ == '__main__':
    main()
