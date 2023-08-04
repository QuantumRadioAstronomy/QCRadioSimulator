import numpy as np
import matplotlib.pyplot as plt

infiles = ['QPIE_eff_vs_imsize_noise10.npy','FRQI_eff_vs_imsize_noise10.npy','QLattice_eff_vs_imsize_noise10.npy']
#infiles = ['QPIE_eff_vs_imsize_noise1.npy','FRQI_eff_vs_imsize_noise1.npy','QLattice_eff_vs_imsize_noise1.npy']


nshot_funcs =[{'func':lambda x: x*x*x*x,     'format':{'c':'orchid','marker':'x'},
               'label':"$N_\mathrm{shots} = N_\mathrm{pix}^2$"}, # N_pix^2
              {'func':lambda x: x*x,         'format':{'c':'blue','marker':'^'},
               'label':"$N_\mathrm{shots} =  N_\mathrm{pix}$"},     # N_pix
              {'func':lambda x: x,           'format':{'c':'orange','marker':'s'},
               'label':"$N_\mathrm{shots} = \sqrt{N_\mathrm{pix}}$"},       # sqrt(N_pix)
              {'func':lambda x: np.log2(x*x),'format':{'c':'red','marker':'v'},
               'label':"$N_\mathrm{shots} = \mathrm{log}_2(N_\mathrm{pix})$"},  # log_2(N_pix)
              ]

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']

f, ax = plt.subplots(3,1,figsize=(6.5,6))
ax = ax.flatten()
plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.1, hspace=0)

x_long = None
for i in range(3):

    with open(infiles[i], 'rb') as f:
        x = np.load(f)
        if i == 0: x_long = x
        y_ref = np.load(f)
        y_ref_err = np.load(f)

        for fn in nshot_funcs:
            fn['y'] = np.load(f)
            fn['yerr'] = np.load(f)
            fn['yrel'] = np.load(f)
            fn['yrelerr'] = np.load(f)

    print(i)
    ax[i].errorbar(x,y_ref, yerr=y_ref_err, c='black', marker='o', label ="Original")
    ax[i].plot([], [], c='gray', ls ='dashed', label = "Relative $\epsilon$")
    ax[i].set_ylim(-0.2,1.25)
    ax[i].set_xlim(7.5**2,70**2)
    for fn in nshot_funcs:
                ax[i].errorbar(x, fn['y'], yerr=fn['yerr'],**fn['format'], label=fn['label'])
                ax[i].errorbar(x, fn['yrel'], yerr=fn['yrelerr'], **fn['format'], ls ='dashed')
    
    ax[i].set_ylabel("Source ID Efficiency")
    ax[i].set_xscale('log')
    ax[i].set_yticks(np.arange(0, 1.25, 0.25))
    
    if i == 2:
        handles, labels = plt.gca().get_legend_handles_labels()
        order=[1,0,2,3,4,5]
        #ax[i].legend(frameon=False)
        ax[i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False) 
        ax[i].set_xlabel("Image size",fontsize=12)

xticks = ["{0}x{0}".format( int(np.sqrt(xi))) for xi in x_long]
print(xticks)
for i in range(3):
    ax[i].set_xticks(x_long)
    ax[i].set_xticklabels(xticks)
    ax[i].tick_params(which='minor', length=0)
ax[0].text(0.01,0.04,"QPIE", transform=ax[0].transAxes,fontsize=12)
ax[1].text(0.01,0.04,"FRQI", transform=ax[1].transAxes,fontsize=12)
ax[2].text(0.01,0.04,"Quantum Lattice", transform=ax[2].transAxes,fontsize=12)

plt.savefig("NSources.pdf")
#plt.savefig("OneSource.pdf")
plt.show()


