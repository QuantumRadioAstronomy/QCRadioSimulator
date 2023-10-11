import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.optimize import minimize

class Simulation:
    def __init__(self, sigma = 1.5, noise_levels= [], imsize = 32, encoding = 'qpie'):
        self.imsize = imsize
        self.noise_levels = noise_levels
        self.sigma = sigma
        
        if encoding == 'qpie' or encoding == 'QPIE':
            self.encoding_type = 'QPIE'
            self.encoding_func = self.make_qpie_img
        elif encoding == 'frqi' or encoding == 'FRQI':
            self.encoding_type = 'FRQI'
            self.encoding_func = self.make_frqi_img
        elif 'lattice' in encoding or 'Lattice' in encoding:
            self.encoding_type = 'QLattice'
            self.encoding_func = self.make_qlattice_img


        if len(noise_levels) == 0: self.noise_level = 0
        else: self.noise_level = noise_levels[0]
        x = np.linspace(0,imsize-1,imsize)
        y = np.linspace(0,imsize-1,imsize)
        self.x, self.y = np.meshgrid(x, y)

    def set_noise_level(self,n):
        self.noise_level = n

    # add noise to an image
    def add_noise(self,img):
        img_shape = img.shape
        img = img.flatten()
        for i,c in enumerate(img):
            img[i] = np.abs(c + np.random.normal(scale=self.noise_level))
        return img.reshape(img_shape)

    # define normalized 2D gaussian
    def gaus2d(self,mx, my, s, a):
        z = a / (2. * np.pi * s * s) * np.exp(-((self.x - mx)**2. / (2. * s**2.) + (self.y - my)**2. / (2. * s**2.)))
        return z

    def calc_dist(self,x1,x2,y1,y2):
        return (x1-x2)**2 + (y1-y2)**2

    # function to fit 2D gaussians to features in the image
    def eval(self,image, nsource, qc = False):
        image = np.copy(image)
        results = []
        for i in range(nsource):
            def cost(args):
                mx, my, s, a = args
                g = self.gaus2d(mx=mx, my= my, s=s, a = a)
                g = g.flatten()
                i = image.flatten()
                diff = (g[np.where(i>0)] - i[np.where(i>0)])**2
                if qc:
                    diff /= i[np.where(i>0)]
                return np.mean( diff**2)

            maxpt = np.unravel_index(image.argmax(), image.shape)
            res = minimize(cost, (maxpt[1],maxpt[0],0.5, 1),
                           bounds= ((0,self.imsize),(0,self.imsize),(self.sigma*0.9,self.sigma*1.1),(0,None)))
            ry, rx = res.x[:2]
            img_fit = self.gaus2d(*res.x)
            results.append([rx,ry,img_fit])
            image -= img_fit
        return results

    def compare_sources(self, s,r):
        distances = []
        for sx,sy in s:
            dsr = [self.calc_dist(sx,ri[0],sy,ri[1]) for ri in r]
            d = np.min(dsr)
            distances.append(np.sqrt(d))
        return distances

    # qpie image is a superposition of position states with amplitude given by the pixel intensity
    # measurement returns one position state 
    def make_qpie_img(self,img, ns):
        N2 = img.shape[0]*img.shape[1]
        img_sampled = np.zeros( N2)

        # create probability amplitudes from color information
        c = (img/np.sum(img)).flatten()

        # measure #shot states. Each shot returns an index
        samples = np.random.choice(np.arange(N2), int(ns), p=c)
        for s in samples:
            img_sampled[s] += 1
        img_sampled = img_sampled.reshape( *img.shape)
        return img_sampled

    # frqi is a superposition of color and position states. 
    # measurement returns one position-color state 
    def make_frqi_img(self,img, ns):
        N2 = img.shape[0]*img.shape[1]
        img_sampled = np.zeros( N2)
        theta = (img/np.sum(img)).flatten()*np.pi/2

        # create the probability amplitudes for the two color states for each position
        c = np.hstack((np.cos(theta),np.sin(theta)))
        c = c/np.sum(c)

        # measure #shot states. Each shot returns and index and a color measurement (0 or 1)
        # 0 corresponds to a state in the first half of the array, 1 to a state in the second half
        samples = np.random.choice(np.arange(N2*2), ns, p=c)
        for s in samples:
            if s >= N2:
                img_sampled[s-N2] += 1

        img_sampled = img_sampled.reshape( *img.shape)
        return img_sampled

    def make_qlattice_img(self,img, ns):
        N2 = img.shape[0]*img.shape[1]
        img_sampled = np.zeros( N2)
        theta = (img/np.sum(img)).flatten()*np.pi/2
        for i in range(N2):
            p = np.array([np.cos(theta[i]),np.sin(theta[i])])
            sample = np.sum(np.random.choice([0,1],  ns, p = p/np.sum(p)))
            img_sampled[i] = sample
        img_sampled = img_sampled.reshape( *img.shape)
        return img_sampled

    def make_obs(self, nsource, nshots,showplot = False):

        # generate mock 'true sources'
        sources = []
        img = np.zeros( (self.imsize,self.imsize))
        for i in range(nsource):
            sx = np.random.randint(2,self.imsize-2)
            sy = np.random.randint(2,self.imsize-2)
            sources.append([sx,sy])
            img[sx,sy] = 1

        # generate mock observation -- telescope has finite resolution
        img_blurred = nd.gaussian_filter(img,self.sigma)
        img_blurred = img_blurred*1/np.max(img_blurred)
        if self.noise_level > 0:
            img_blurred = self.add_noise(img_blurred)

        # run reconstruction
        res1 = self.eval(img_blurred,nsource)
        img_res1 = np.array([r[2] for r in res1])
        img_fit1 = np.sum(img_res1, axis = 0)
        sources_fit1 = [ [r[0],r[1]] for r in res1]
        d1 = self.compare_sources(sources, sources_fit1)

        # generate mock QC sampling
        imgs_qc = {}
        res_qc = {}
        #sources_qc = {}
        d_qc = {}
        for ns in nshots:
            #img_sampled = self.make_qpie_img(img_blurred, ns)
            img_sampled = self.encoding_func(img_blurred, ns)
            
            # run reconstruction
            res2 = self.eval(img_sampled,nsource)
            img_res2 = np.array([r[2] for r in res2])
            img_fit2 = np.sum(img_res2, axis = 0)
            sources_fit2 = [ [r[0],r[1]] for r in res2]

            imgs_qc[ns] = img_sampled
            res_qc[ns] = img_fit2
            #sources_qc[ns] = sources_fit2
            d_qc[ns] = self.compare_sources(sources, sources_fit2)

        if showplot:
            fig, axs = plt.subplots(2,2+len(nshots), figsize=(12, 5))
            for a in axs.flatten(): a.set_axis_off()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=.3, hspace = 0.3)

            axs[0,0].imshow(img, cmap = "cubehelix")
            axs[0,0].set_title("True sources")
            
            axs[0,1].imshow(img_blurred, cmap = "cubehelix")
            axs[0,1].set_title("Observed image")
            axs[1,1].imshow(img_fit1, cmap = "cubehelix")
            axs[1,1].set_title("Fit to observed image")

            for ni, ns in enumerate(nshots):
                axs[0,2+ni].imshow(imgs_qc[ns], cmap = "cubehelix")
                axs[0,2+ni].set_title("{1}\nreconstructed with\n{0} shots".format(ns,self.encoding_type))
                axs[1,2+ni].imshow(res_qc[ns], cmap = "cubehelix")
                axs[1,2+ni].set_title("Fit to reconstructed\nquantum image")

            plt.savefig("{1}_example_noise{0:.0f}.pdf".format(self.noise_level*100,self.encoding_type))
            plt.show()

        return d1,d_qc

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



