
import  matplotlib.pyplot as plt
import seaborn as sns

#from matplotlib import rc,rcParams
def plot_error_distribution(result, dts_name):
    margin_left = 0.05
    margin_right= 0.99
    margin_top = 0.99
    margin_bottom = .23
    #plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=margin_right, top=margin_top)

    plt_width = 38
    plt_heigth = 18
    #plt.figure(figsize=(plt_width,plt_heigth))

    plot_rotation = 75
    axis_font_size = 50
    #plt.xticks(rotation =plot_rotation ,fontsize = axis_font_size)
    #plt.yticks(fontsize = axis_font_size)

    labels_size = 60    
    #ax.set_xlabel("Qu


    #print(order2)
    with sns.axes_style("whitegrid"), sns.color_palette('Spectral', result['quantifier'].nunique()):

        #plt.figure(figsize=(38,18))
        plt.figure(figsize=(plt_width,plt_heigth))
        #plt.margins(0.01, tight=True)
        plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=margin_right, top=margin_top)
        #plt.subplots_adjust(left=.05, bottom=.23, right=.99, top=.99)
        ax=sns.boxplot(data=result, x='quantifier', y='error') 
        #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"07"})

        #plt.xticks(rotation =75 ,fontsize = 50)
        #plt.yticks(fontsize = 50)
        plt.xticks(rotation =plot_rotation ,fontsize = axis_font_size)
        plt.yticks(fontsize = axis_font_size)

        #ax.set_xlabel("Quantifiers",fontsize=60)
        #ax.set_ylabel("Avg. ranking",fontsize=60)
        ax.set_xlabel("Quantifiers",fontsize=labels_size)
        ax.set_ylabel("AE",fontsize=labels_size)
        ax.set_title( "Dataset: "+ dts_name,fontsize=45)

        #plt.legend( bbox_to_anchor=(1.15, 1),loc='upper right')



    #ax.figure.savefig(folder + '/Train_prop_%f'% (train_prop) + 'Test_size %d'% (test_size) + '.png', format="PNG" )
    #ax.figure.savefig(folder + '/Overall_Rank'  + '.pdf', format="PDF" )
    plt.show()


#from matplotlib import rc,rcParams
def plot_rank(result, dts_name):
    margin_left = 0.05
    margin_right= 0.99
    margin_top = 0.99
    margin_bottom = .23
    #plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=margin_right, top=margin_top)

    plt_width = 38
    plt_heigth = 18
    #plt.figure(figsize=(plt_width,plt_heigth))

    plot_rotation = 75
    axis_font_size = 50
    #plt.xticks(rotation =plot_rotation ,fontsize = axis_font_size)
    #plt.yticks(fontsize = axis_font_size)

    labels_size = 60    
    #ax.set_xlabel("Qu


    #print(order2)
    with sns.axes_style("whitegrid"), sns.color_palette('Spectral', result['quantifier'].nunique()):

        #plt.figure(figsize=(38,18))
        plt.figure(figsize=(plt_width,plt_heigth))
        #plt.margins(0.01, tight=True)
        plt.subplots_adjust(left=margin_left, bottom=margin_bottom, right=margin_right, top=margin_top)
        #plt.subplots_adjust(left=.05, bottom=.23, right=.99, top=.99)
        ax=sns.boxplot(data=result, x='quantifier', y='rank') 
        #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"07"})

        #plt.xticks(rotation =75 ,fontsize = 50)
        #plt.yticks(fontsize = 50)
        plt.xticks(rotation =plot_rotation ,fontsize = axis_font_size)
        plt.yticks(fontsize = axis_font_size)

        #ax.set_xlabel("Quantifiers",fontsize=60)
        #ax.set_ylabel("Avg. ranking",fontsize=60)
        ax.set_xlabel("Quantifiers",fontsize=labels_size)
        ax.set_ylabel("Rank",fontsize=labels_size)
        #ax.set_title( "Dataset: "+ dts_name,fontsize=45)

        #plt.legend( bbox_to_anchor=(1.15, 1),loc='upper right')



    #ax.figure.savefig(folder + '/Train_prop_%f'% (train_prop) + 'Test_size %d'% (test_size) + '.png', format="PNG" )
    #ax.figure.savefig(folder + '/Overall_Rank'  + '.pdf', format="PDF" )
    plt.show()