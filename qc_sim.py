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