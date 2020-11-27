import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file = 'tour194'

show_iteration_plots = False
show_heatmap_plot = True

filename_blueprint = "grid_search_results/" + file + '/' + "r0123456" + 'iter={}_stopcrit{}_lambda={}_alpha={}_k={}_' + file + '.csv'

# Iteration plots
if show_iteration_plots:
    if file == 'tour29':
        amountOfiterations = 3000
        stopIteratingAfter = 500
        k_elimination = 5
        gs_lam = [500]
        gs_alpha = [0.01, 0.03, 0.04, 0.05, 0.075, 0.08, 0.1, 0.25, 0.5]
        gs_k = [1, 2, 3, 4, 5, 6, 10]
    if file == 'tour194':
        amountOfiterations = 4000
        stopIteratingAfter = 300
        k_elimination = 5
        gs_lam = [50, 75, 100, 200, 500, 600]
        gs_alpha = [0.04]
        gs_k = [4]

    plot_alpha_experiments = False
    plot_k_experiments = False
    plot_lam_experiments = True

    alpha_experiments = []
    alpha_experiment_k = 2
    for l in gs_lam:
        for a in gs_alpha:
            alpha_experiments.append({"lam": l, "alpha": a, "k": alpha_experiment_k})


    k_experiments = []
    k_experiment_alpha = 0.4
    for l in gs_lam:
        for k in gs_k:
            k_experiments.append({"lam": l, "alpha": k_experiment_alpha, "k": k})


    lam_experiments = []
    lam_experiments_alpha = 0.04
    lam_experiments_k = 4
    for l in gs_lam:
        lam_experiments.append({"lam": l, "alpha": lam_experiments_alpha, "k": lam_experiments_k})


    if plot_lam_experiments:
        for e in lam_experiments:
            lam = e["lam"]
            alpha = e["alpha"]
            k = e["k"]
            filename = filename_blueprint.format(str(amountOfiterations), str(stopIteratingAfter), str(lam), str(alpha),
                                                 str(k))
            x = []
            y = []
            file_length = 0
            try:
                with open(filename, 'r') as csvfile:
                    plots = csv.reader(csvfile, delimiter=',')
                    count = 0
                    for row in plots:
                        if count < 2:
                            count += 1
                            continue
                        count += 1
                        # print(row[0])
                        x.append(float(row[0]))
                        y.append(float(row[2]))
                print("{} {}".format(lam, y[-1]))
                plt.plot(x, y, label='lambda=' + str(lam))
                plt.xlabel('iterations')
                plt.ylabel('mean fitness')
                plt.title(file + ': Mean fitness vs iterations\nalpha={} k={}'.format(alpha, k))
                plt.legend()
            except:
                print("File not found, skipping...")

        plt.figure()

    if plot_alpha_experiments:
        for e in alpha_experiments:
            lam = e["lam"]
            alpha = e["alpha"]
            k = e["k"]
            filename = filename_blueprint.format(str(amountOfiterations), str(stopIteratingAfter), str(lam), str(alpha), str(k))
            x= []
            y= []
            file_length = 0
            try:
                with open(filename,'r') as csvfile:
                    plots = csv.reader(csvfile, delimiter=',')
                    count = 0
                    for row in plots:
                        if count < 2:
                            count += 1
                            continue
                        count += 1
                        #print(row[0])
                        x.append(float(row[0]))
                        y.append(float(row[2]))
                print("{} {}".format(k, y[-1]))
                plt.plot(x,y, label='alpha='+str(alpha))
                plt.xlabel('iterations')
                plt.ylabel('mean fitness')
                plt.title(file + ': Mean fitness vs iterations\nlambda={} k={}'.format(lam, k))
                plt.legend()
            except:
                print("File not found, skipping...")

        plt.figure()

    if plot_k_experiments:
        for e in k_experiments:
            lam = e["lam"]
            alpha = e["alpha"]
            k = e["k"]
            filename = filename_blueprint.format(str(amountOfiterations), str(stopIteratingAfter), str(lam), str(alpha), str(k))
            x= []
            y= []
            try:
                with open(filename,'r') as csvfile:
                    plots = csv.reader(csvfile, delimiter=',')
                    count = 0
                    for row in plots:
                        if count < 2:
                            count += 1
                            continue
                        count += 1
                        #print(row[0])
                        x.append(float(row[0]))
                        y.append(float(row[2]))
                print("{} {}".format(alpha, y[-1]))
                plt.plot(x,y, label='k='+str(k))
                plt.xlabel('iterations')
                plt.ylabel('mean fitness')
                plt.title(file + ': Mean fitness vs iterations\nlambda={} alpha={}'.format(lam, alpha))
                plt.legend()
            except:
                print("File not found, skipping...")

    plt.show()




if show_heatmap_plot:
    if file=='tour29':
        amountOfiterations = 3000
        stopIteratingAfter = 500
        k_elimination = 5
        gs_lam = [500]
        gs_alpha = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.08, 0.1, 0.25, 0.5]
        gs_k = [1, 2, 3, 4, 5, 6, 10]
    if file=='tour194':
        amountOfiterations = 4000
        stopIteratingAfter = 300
        k_elimination = 5
        gs_lam = [75]
        gs_alpha = [0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        gs_k = [1, 2, 3, 4, 5]
    if file=='tour929':
        amountOfiterations = 4000
        stopIteratingAfter = 300
        k_elimination = 5
        gs_lam = [50]
        gs_alpha = [0.1, 0.4, 0.5, 0.6, 0.8]
        gs_k = [1, 3, 5]

    # Grid search heatmap
    experiments = []
    for l in gs_lam:
        for a in gs_alpha:
            for k in gs_k:
                experiments.append({"lam":l, "alpha":a, "k":k})

    # get mean fitness of last iterations, x-axis alpha (columns), y-axis k (rows), heat score = value (mean fitness of last iter)
    best_gs_mean_fitnesses = np.array([[0]*len(gs_alpha)] * len(gs_k))
    # 7x6 matrix

    for e in experiments:
        lam = e["lam"]
        alpha = e["alpha"]
        k = e["k"]
        filename = filename_blueprint.format(str(amountOfiterations), str(stopIteratingAfter), str(lam), str(alpha), str(k))
        file_length = 0

        try:
            with open(filename,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                file_length = sum(1 for row in plots)

            with open(filename,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                count = 0
                for row in plots:
                    if count < 2:
                        count += 1
                        continue
                    count += 1
                    #print(row[0])
                    if count == file_length:
                        print("{} x {}".format(gs_k.index(k), gs_alpha.index(alpha)))
                        best_gs_mean_fitnesses[gs_k.index(k)][gs_alpha.index(alpha)] = float(row[2])
                        for i in best_gs_mean_fitnesses:
                            print(i)
                        print()
        except:
            print("File not found for this combination of paramters, skipping...")


    param_grid = {'alpha': gs_alpha, 'k': gs_k}
    print(param_grid)

    # Set unknown values (no experiment done because assume high value) to the maximum value to not mess up scale of color legend
    worst_mean_fitness = best_gs_mean_fitnesses.max()
    value_to_fill_in_unknown = worst_mean_fitness
    for i in range(len(best_gs_mean_fitnesses)):
        for j in range(len(best_gs_mean_fitnesses[i])):
            if best_gs_mean_fitnesses[i][j] == 0:
                best_gs_mean_fitnesses[i][j] = value_to_fill_in_unknown

    # Get best params combination
    best_params = {'alpha':None, 'k':None}
    best_mean_fitness = best_gs_mean_fitnesses.min()
    for i in range(len(best_gs_mean_fitnesses)):
        for j in range(len(best_gs_mean_fitnesses[i])):
            if best_gs_mean_fitnesses[i][j] == best_mean_fitness:
                best_params['alpha'] = gs_alpha[j]
                best_params['k'] = gs_k[i]
    print("Best parameters: alpha={} k={}".format(best_params['alpha'], best_params['k']))



    # make heatmap
    import pandas as pd

    pvt = pd.DataFrame(best_gs_mean_fitnesses, columns = gs_alpha, index = gs_k)
    ax = sns.heatmap(pvt, annot=True, fmt="d")

    plt.xlabel('alpha')
    plt.ylabel('k')
    plt.title(file + ': Heatmap of Mean fitness for different alpha and k with lambda={}\nBest value for alpha={} and k={}'.format(gs_lam[0], best_params['alpha'], best_params['k']))

    plt.show()