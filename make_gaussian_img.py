import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.optimize import minimize
from qc_sim import Simulation



def plot_single_simulation(nsource = 5, nshots = [10, 100], noise = 0, encoding = 'qpie'):
    mysim = Simulation(noise_levels = [noise], encoding = encoding)
    mysim.make_obs(nsource = nsource, nshots = nshots, showplot=True)

def run_many_simulations(nsim = 10, nsource = 5, nshots = [10, 100], noise = 0,  imsize = 32, encoding = 'QPIE', showplot = True):
    mysim = Simulation(noise_levels = [noise], imsize = imsize, encoding = encoding)
    d_cc = []
    d_qc = {}
    for ns in nshots: d_qc[ns] = []
    for trial in range(nsim):
        d1, d2 = mysim.make_obs(nsource = nsource, nshots = nshots, showplot = (showplot if trial == 0 else False))
        d_cc.extend(d1)
        for ns in nshots: d_qc[ns].extend(d2[ns])

    print("#======\nNoise",noise)
    d = np.array(d_cc)
    print("orig", len(d[d < 1.5])*1./len(d) )
    for ns in nshots[::-1]:
        d = np.array(d_qc[ns])
        print(ns, len(d[d < 1.5])*1./len(d) )
    

    if showplot:

        f, ax = plt.subplots(figsize=(7, 5))

        qc_color = {10:'red', 100:'orchid', 1000:'navy', 10000:'dodgerblue'}
        qc_style   = {10:{'ls':'dotted','lw':1.5, 'histtype':'step',  'edgecolor':qc_color[10]},
                     100:{'ls':'dashed','lw':1.5, 'histtype':'step', 'edgecolor':qc_color[100]},
                    1000:{'ls':'solid','lw':1.5,  'histtype':'step', 'edgecolor':qc_color[1000]},
                   10000:{'histtype':'step', 'hatch':'////','color':qc_color[10000], 'edgecolor':qc_color[10000]}}

        bins = [0,1,2,3,4,5,6,7,8,9,10]
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
        orig_label = "Original, avg. dist.$= {0:.2f}\pm{1:.2f}$ pix".format(np.mean(d_cc),np.std(d_cc))
        plt.hist(d_cc, bins = bins, color = 'lightskyblue', alpha = 1.0, histtype = 'stepfilled', label = orig_label)

        for ns in nshots[::-1]:
            d = np.array(d_qc[ns])
            hist_label = "{3} Reco. w/ {2} shots, avg. dist. $= {0:.2f}\pm{1:.2f}$ pix".format(np.mean(d),np.std(d),ns,encoding)
            plt.hist(d, bins = bins, **qc_style[ns], label = hist_label)


        text_sim = "Simulated image with {0} sources, beam size of $\sigma={1}$ pixels".format(nsource,mysim.sigma)
        text_noise = ", Gaussian noise {0:.0f}% of max".format(noise*100) if noise >0 else ''
        plt.text(0.5,1.05,"{0}{1}".format(text_sim,text_noise), transform=ax.transAxes, ha = 'center')

        plt.xlabel("Distance from reconstructed to true source [pixels]")
        plt.legend(frameon=False)
        plt.savefig("{1}_summary_noise{0:.0f}.pdf".format(noise*100, encoding))
        plt.show()
    return d_cc, d_qc 


def get_eff_and_err(d, th = 1.5):
    d = np.array(d)
    N = len(d)
    k = len(d[d < 1.5])
    eff = k*1./N

    # using Poisson efficiency error
    err = eff*np.sqrt( (1/k if k >0 else 1) + 1/N)

    return eff, err

def eff_vs_noise(SNR, nshots, encoding = 'QPIE'):

    eff_cc = []
    err_cc = []
    eff_qc = {}
    err_qc = {}
    for ns in nshots:
        eff_qc[ns] = []
        err_qc[ns] = []

    for snr in SNR:
        ni = 1./snr
        d_cc, d_qc = run_many_simulations(100,nshots = nshots, noise = ni, encoding = encoding, showplot = False)

        eff, err = get_eff_and_err(d_cc)
        eff_cc.append(eff)
        err_cc.append(err)

        for ns in nshots:
            eff, err = get_eff_and_err(d_qc[ns])
            eff_qc[ns].append(eff)
            err_qc[ns].append(err)

    f, ax = plt.subplots()
    print( len(eff_cc), len(err_cc))
    print (eff_cc, err_cc)
    plt.errorbar(SNR,eff_cc, yerr = err_cc,c='lightskyblue',marker='o',label ="Original")
    qc_color = {10:'red', 100:'orchid', 1000:'navy', 10000:'dodgerblue'}
    qc_style   = {10:{'marker':'s','ls':'dotted'},
                         100:{'marker':'v','ls':'dashed'},
                        1000:{'marker':'^','ls':'solid'},
                       10000:{'marker':'o','markerfacecolor':'none'}}
    for ns in nshots[::-1]:
        plt.errorbar(SNR,eff_qc[ns], yerr = err_qc[ns], c=qc_color[ns], **qc_style[ns], label ="{1} Reco. w/ {0} shots".format(ns, encoding))
    plt.xlabel("Signal-to-Noise Ratio")
    plt.ylabel("Source ID Efficiency")
    ax.set_ylim(-0.3,1.0)
    plt.xscale('log')
    plt.legend(frameon=False, ncol=2)
    plt.savefig(encoding+ "_eff_vs_noise2.pdf")
    plt.show()

def eff_vs_imsize(imsize, noise = 0.1, encoding = 'qpie', nsource = None):

    eff_cc = []
    err_cc = []
    eff_qc = [[],[],[],[],[]]
    err_qc = [[],[],[],[],[]]

    nshot_funcs =[{'func':lambda x: x*x*x*x,     'format':{'c':'orchid','marker':'P'},
                   'label':encoding+", $N_\mathrm{shots} = N_\mathrm{pix}^2$"}, # N_pix^2
                  {'func':lambda x: x*x,         'format':{'c':'blue','marker':'^'},
                   'label':encoding+", $N_\mathrm{shots} =  N_\mathrm{pix}$"},     # N_pix
                  {'func':lambda x: x,           'format':{'c':'orange','marker':'s'},
                   'label':encoding+", $N_\mathrm{shots} = \sqrt{N_\mathrm{pix}}$"},       # sqrt(N_pix)
                  {'func':lambda x: np.log2(x*x),'format':{'c':'red','marker':'v'},
                   'label':encoding+", $N_\mathrm{shots} = \mathrm{log}_2(N_\mathrm{pix})$"},  # log_2(N_pix)
                  ]

    for ims in imsize:
        if nsource == None:
            nsour = int(ims*ims*4/(32*32))+1
        else:
            nsour = nsource
        nshots = [ int(fn['func'](ims)) for fn in nshot_funcs]
        nsim = 100
        if 'ttice' in encoding and ims > 32: nsim = 10
        d_cc, ds_qc = run_many_simulations(nsim, nsource = nsour,
                                                nshots = nshots,
                                                noise = noise, imsize = ims,
                                                encoding = encoding,
                                                showplot = False)

        eff, err = get_eff_and_err(d_cc)
        eff_cc.append(eff)
        err_cc.append(err)

        for i, ns, in enumerate(nshots):
            eff, err = get_eff_and_err(ds_qc[ns])
            eff_qc[i].append(eff)
            err_qc[i].append(err)

    outname = "{1}_eff_vs_imsize_noise{0:.0f}".format(noise*100, encoding)
    with open(outname +'.npy', 'wb') as npf:
        f, ax = plt.subplots(figsize=(5,5))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
        ax.set_ylim(-0.6,1.2)
        plt.errorbar(imsize**2,eff_cc, yerr=err_cc, c='black', marker='o', label ="Original")
        np.save(npf, imsize**2)
        np.save(npf, eff_cc)
        np.save(npf, err_cc)
        plt.plot([], [], c='gray', ls ='dashed', label = encoding+" relative $\epsilon$")
        #plt.plot(ax.get_xlim(), [0,0], c='gray', ls ='dotted', lw = '0.5')
        for i, fn in enumerate(nshot_funcs):
            plt.errorbar(imsize**2, eff_qc[i], yerr=err_qc[i],**fn['format'], label=fn['label'])
            r = np.array(eff_qc[i])/np.array(eff_cc)
            rerr = r*np.sqrt(np.array(err_cc)**2 + np.array(err_qc[i])**2)
            plt.errorbar(imsize**2, r, yerr=rerr, **fn['format'], ls ='dashed')

            np.save(npf, eff_qc[i])
            np.save(npf, err_qc[i])
            np.save(npf, r)
            np.save(npf, rerr)




        plt.xlabel("Image size")
        plt.ylabel("Source ID Efficiency")
        plt.xscale('log')
        plt.xticks(imsize**2, ["{0}x{0}".format(ims) for ims in imsize])
        plt.title("Simulated images with SNR = {0:.0f}".format(1/noise))
        
        plt.legend(frameon=False, ncol=2, loc='lower center')
        plt.savefig(outname + ".pdf")
        plt.show()


from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']

nshots = [10, 100, 1000, 10000]
#imsizes = np.array([8,16,32,64])
imsizes = np.array([8,16,32,64])

#plot_single_simulation(nshots = nshots, noise = 0.1, encoding = "FRQI")
#plot_single_simulation(nshots = nshots, noise = 0.1, encoding = "QLattice")
#run_many_simulations(  nshots = nshots, noise = 0.01, nsim = 100)
#run_many_simulations(  nshots = nshots, noise = 0.1,  nsim = 100) #0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

#eff_vs_noise(noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], nshots = nshots, encoding = 'QPIE')
#eff_vs_noise(SNR = [100, 50, 20, 10, 5, 4 ,3 ,2 ,1], nshots = nshots, encoding = 'QPIE')
#eff_vs_imsize(np.array([8,16,32,64]),noise=0.1, encoding = 'QPIE')
eff_vs_imsize(np.array([8,16,32,64]),noise=0.1, encoding = 'FRQI')
eff_vs_imsize(np.array([8,16,32]),noise=0.1, encoding = 'QLattice')

#eff_vs_imsize(np.array([8,16,32,64]),noise=0.01, encoding = 'QPIE', nsource = 1)
#eff_vs_imsize(np.array([8,16,32,64]),noise=0.01, encoding = 'FRQI', nsource = 1)
#eff_vs_imsize(np.array([8,16,32]),noise=0.01, encoding = 'QLattice', nsource = 1)



