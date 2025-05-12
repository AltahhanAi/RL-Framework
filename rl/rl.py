
'''
   Author: Abdulrahman Altahhan, 2025.
   version: 3.3

    This library of functionality in RL that aims for simplicity and general insight into how algorithms work, these libraries 
    are written from scratch using standard Python libraries (numpy, matplotlib etc.).
    Please note that you will need permission from the author to use the code for research, commercially or otherwise.
'''


'''
    This is a full-fledged RL library. The assumption is a tabular representation.
    However, these classes can be easily extended to linear and non-linear function
    approximation as we shall see in other libraries.
    
    We start by defining a base class called MRP(Markov Reward process). This class focuses
    on predicting the value function (discounted sum of rewards). This is a beneficial 
    method of identifying the effect of actions (control). The typical problem that allows us to test
    prediction only is a random walk problem.
    
    We then define another base class called MDP(Markov Decision Process). This class focuses
    on control by implementing policy changes/improvements capabilities. It will allow us to change the 
    Q table, which indirectly influences the probabilities of taking certain actions. The Q table specifies
    the usefulness (expected discounted sum of rewards) of taking a certain action in a certain state.

    We finally define another base class called PG(Policy Gradient). This class focuses on incorporating
    both value prediction and policy improvement at the same time. This can be very effective for practical
    problems, including robotics. PG methods attempt to directly learn the best probability function of
    taking a certain action in a certain state in order to come up with the best policy. So it directly
    learns the best policy instead of indirectly learning a Q function that, in turns, influences the policy.
    
    predict value function: MC, TD, TDf...
    control value function: MCC, Sarsa, n-step Saras, Q-learning DDQ...
    control policy gradient: REINFORCE, Actor-critic...
'''

# ====================================================================================================================
from env.grid import *
from rl.rlselect import *
from rl.dp import *
import pickle
from pathlib import Path
# ==============================================Base class for prediction =============================================
'''
    all other classes will inherit from this class.
'''
class MRP:
    
    def __init__(self, env=randwalk(), γ=1, α=.1, v0=0, episodes=100, view=1,
                 store=False, # Majority of methods are pure one-step online and no need to store episodes trajectories 
                 max_t=2000, seed=None, visual=False, underhood='',
                 last=10, print_=False, self_path='experiment.pkl'):
        # hyper parameters
        self.env = env
        self.γ = γ
        self.α = α # average methods(like MC1st) do not need this but many other methods (like MCα) do
        self.v0 = v0
        self.episodes = episodes
        self.store = store
        self.max_t = max_t
        self.visual = visual
        self.view = view
        self.underhood = underhood
        self.last = last
        self.print_ = print_
        self.self_path = self_path  # path and name of the pickle file
        
        # reference to two important functions
        self.policy = self.stationary
        self.step = self.step_a
        # we might want to skip a step
        self.skipstep = False
        
        nA = self.env.nA
        self.As = list(range(nA))
        self.pAs = [1/nA]*nA
        
        # useful to repeate the same experiement
        self.seed(seed)
        # to protect interact() in case of no training 
        self.ep = -1
        
    # set up important metrics
    def init_metrics(self):
        self.Ts = np.zeros(self.episodes, dtype=np.uint32)
        self.Rs = np.zeros(self.episodes)
        self.Es = np.zeros(self.episodes)
    
    def extend_metrics(self):
        if len(self.Ts)>=self.episodes: return # no need to resize if size is still sufficient
        self.Ts = np.resize(self.Ts, self.episodes)
        self.Rs = np.resize(self.Rs, self.episodes)
        self.Es = np.resize(self.Es, self.episodes)
        
    # set up the V table
    def init_(self):
        self.V = np.ones(self.env.nS)*self.v0

    # useful for inheritance, gives an expected return (value) for state s
    def V_(self, s=None): 
        return self.V  if s is None else self.V[s]
    
    def seed(self, seed=None, **kw):
        if seed is not None: np.random.seed(seed); random.seed(seed)
    #-------------------------------------------buffer related-------------------------------------------------
    # The buffer get reinitialised by reinitialising t only but we have to be careful not to exceed t+1 at any time
    def allocate(self): 
        if not self.store: return
        self.r = np.zeros(self.max_t)
        self.s = np.ones(self.max_t, dtype=np.uint32)*(self.env.nS+10) # states are indices:*(nS+10)for debugging 
        self.a = np.ones(self.max_t, dtype=np.uint32)*(self.env.nA+10) # actions are indices:*(nA+10)for debugging       
        self.done = np.zeros(self.max_t, dtype=bool)
    
    def store_(self, s=None,a=None,rn=None,sn=None,an=None, done=None, t=0):
        if not self.store: return
        if s  is not None: self.s[t] = s
        if a  is not None: self.a[t] = a
        if rn is not None: self.r[t+1] = rn
        if sn is not None: self.s[t+1] = sn
        if an is not None: self.a[t+1] = an
        if done is not None: self.done[t+1] = done
    
    def stop_ep(self, done):
        return done or (self.t+1 >= self.max_t-1) # goal reached or storage is full
    
    # ------------------------------------ experiments related --------------------------------------------
    def stop_exp(self):
        if self.stop_early(): print('experience stopped at episode %d'%self.ep); return True
        return self.ep >= self.episodes - 1

    #----------------------------------- steps as per the algorithm style --------------------------------
    def step_0(self):
        s = self.env.reset()                                 # set env/agent to the start position
        a = self.policy(s)
        return s,a
    
    # accomodates Q-learning and V style algorithms
    def step_a(self, s,_, t):                          
        if self.skipstep: return 0, None, None, None, True
        a = self.policy(s)
        sn, rn, done, _ = self.env.step(a)

        # we added s=s for compatibility with deep learning
        self.store_(s=s, a=a, rn=rn, sn=sn, done=done, t=t)

        # None is returned for compatibility with other algorithms
        return rn,sn, a,None, done

    # accomodates Sarsa style algorithms
    def step_an(self, s,a, t):                          
        if self.skipstep: return 0, None, None, None, True
        sn, rn, done, _ = self.env.step(a)
        an = self.policy(sn)

        # we added s=s for compatibility with deep learning later
        self.store_(s=s, a=a, rn=rn, sn=sn, an=an, done=done, t=t)
        return rn,sn, a,an, done

    #------------------------------------  online learning and interaction --------------------------------
    def interact(self, train=True, resume=False, save_ep=False, 
                 overwrite=False, episodes=None, env=None, **kw):
        if episodes is not None: self.episodes=episodes
        if env is not None: self.env = env
        if train and not resume: # train from scratch or resume training
            self.init_()
            self.init()                                        # user defined init() before all episodes
            self.init_metrics()
            self.allocate()
            self.selfsave(overwrite) if save_ep else None
            self.plot0()                                       # useful to see initial V values
            self.seed(**kw)
            self.ep = -1  #+ (not train)*(self.episodes-1)
            self.t_ = 0                                        # steps counter for all episodes
        if resume:
            self.extend_metrics()
        # try:
            #for self.ep in range(self.episodes):
        while not self.stop_exp():
            self.ep += 1
            self.t  = -1                                    # steps counter for curr episode
            self.Σr = 0
            done = False
            #print(self.ep)
            # initial step
            s,a = self.step_0()
            self.step0()                                    # user defined init of each episode
            # an episode is a set of steps, interact and learn from experience, online or offline.
            while not self.stop_ep(done):
                #print(self.t_)

                # take one step
                self.t += 1
                self.t_+= 1

                rn,sn, a,an, done = self.step(s,a, self.t)  # takes a step in env and store tarjectory if needed
                self.online(s, rn,sn, done, a,an) if train else None # to learn online, pass a one step trajectory

                self.Σr += rn
                self.rn = rn
                s,a = sn,an

                # render last view episodes, for games ep might>episodes
                if self.visual and self.episodes > self.ep >= self.episodes-self.view: self.render(**kw)

            # to learn offline and plot episode
            self.metrics()
            self.offline() if train else None
            self.plot_ep()
            self.selfsave(overwrite) if save_ep else None  # saves object in a pickle file for retrieval in case of crash
            self.env.stop() if hasattr(self.env, 'stop') else None

        # except:
        # print(f"Either learning interrupted or an error occurred:, at state {s}")
        # try: self.env.stop()
        # finally: pass
        # finally:
            # plot experience   
            # self.plot_exp(**kw)
        return self
    # ------------------------------------- policies types 易-----------------------------------

    def stationary(self, *args):
        #return choice(self.As, 1, p=self.pAs)[0] # this gives better experiements quality but is less efficient
        return choices(self.As, weights=self.pAs, k=1)[0] if self.env.nA!=2 else np.random.binomial(1, 0.5)
    
    #---------------------------------------perfromance metrics ------------------------------
    def metrics(self):
        # we use %self.episodes so that when we use a different criterion to stop_exp() code will run
        self.Ts[self.ep%self.episodes] = self.t+1
        self.Rs[self.ep%self.episodes] = self.Σr
        self.Es[self.ep%self.episodes] = self.Error()
        
        if self.print_: print(self)
    
    # retruns indexes of last n elements that spans two edges of an array, i is current index
    # also it retruns the element of current index
    def circular_n(self, A, i, n):
        N = len(A)
        i, n, inds = i%N, min(i+1, n), np.ones(N, dtype=bool)              
        inds[i+1: N+1 - (n-i)] = False  # turn off indexes that we do not want, to deal with circular indexes
        return A[inds][-n:], A[i]
    
    def __str__(self):
        # mean works regardless of where we stored the episode metrics (we use %self.episodes)     
        Rs, R = self.circular_n(self.Rs, self.ep, self.last) # this function is defined above
        metrics = 'step %d, episode %d, r %.2f, mean r last %d ep %.2f, ε %.2f'
        values = (self.t_, self.ep, R, self.last, Rs.mean().round(2), round(self.ε, 2))
        return metrics%values

    #------------------------functions that can be overridden in the child class-----------------
    def init(self):
        pass
    def step0(self):
        pass
    def Error(self):
        return 0
    def stop_early(self):
        return False
    def plot0(self):
        pass
    def plot_t(self):
        pass
    def plot_ep(self):
        pass
    def plot_exp(self, *args):
        pass
    def offline(self):
        pass
    def online(self,*args):
        pass
    
    # infrastructure for saving an object, useful for long experiments that can crash
    def selfsave(self, overwrite):
        env = self.env
        self.env = None # execlude the env as it will cause issues when dealing with ros
        if Path(self.self_path).exists() and not overwrite:
            print(f"Warning: {self.self_path} already exists and you set overwrite the pickl to False, so we are existing.")
            return
        try:
            with open(self.self_path, "wb") as f: pickle.dump(self, f)
            # print(f"Object saved to {self.self_path}")
        except: print(f'could not save the file {self.self_path}')
        finally: self.env = env
        
    @classmethod
    def selfload(cls, self_path=None):
        if self_path is None: self_path = cls.self_path# use default path if it was not provide 
        try:
            with open(self_path, "rb") as f: obj = pickle.load(f)
            print(f"Object restored from {self_path}")
        except: print(f'could not load the file {self_path}')
        return obj
    
    #---------------------------------------visualise ✍️----------------------------------------
    # overload the env render function
    def render(self, rn=None, label='', **kw):
        if rn is None: rn=self.rn
        param = {'V':self.V_()} if self.underhood=='V' else {}
        self.env.render(**param, 
                        label=label+' reward=%d, t=%d, ep=%d'%(rn, self.t+1, self.ep+1), 
                        underhood=self.underhood, 
                        **kw)

# -----------------------------random walk visualisation convenience extension ---------------------------
'''
    Adding some visualisation capabilities for random walk problem.
    We use the same name to avoid using lots of different names and to keep our code simple.
'''

class MRP(MRP):
    
    def __init__(self, plotV=False,  plotT=False, plotR=False, plotE=False, animate=False, Vstar=None, **kw):
        super().__init__(**kw)
        
        # visualisation related
        self.plotT = plotT
        self.plotR = plotR
        self.plotE = plotE
        self.plotV = plotV 
        self.animate = animate
        self.eplist = []
        
        nS = self.env.nS
        self.Vstar = Vstar if Vstar is not None else self.env.Vstar
    #------------------------------------------- metrics -----------------------------------------------  
    # returns RMSE but can be overloaded if necessary
    # when Vstar=0, it shows how V is evolving via training 
    def Error(self):
        if self.Vstar is None: return 0
        return np.sqrt(np.mean(((self.V_() - self.Vstar)[1:-1])**2)) #if self.Vstar is not None else 0
    
    #--------------------------------------------visualise ✍️----------------------------------------------

    def plot0(self):
        if self.plotV: self.plot_V(); plt.show()
        
    def plot_exp(self, label='', **kw):
        self.plot_ep(animate=True, plot_exp=True, label=label)
        
    def plot_ep(self, animate=None, plot_exp=False, label=''): 
        if len(self.eplist)< self.episodes: self.eplist.append(self.ep+1)
            
        if animate is None: animate = self.animate
        if not animate: return
        frmt='.--'if not plot_exp or self.ep==0 else '--'

        if self.visual: 
            if self.ep==self.episodes-1: self.render(animate=False) # shows the policy 
            else:                        self.env.render(animate=False) 
        if self.plotV:  self.plot_V(ep=self.ep+1)        
        
        i=2
        for plot, ydata, label_ in zip([self.plotT, self.plotR, self.plotE], 
                                      [self.Ts,    self.Rs,    self.Es   ], 
                                      ['steps   ', 'Σrewards', 'Error   ']):
            if not plot: continue
            plt.subplot(1,3,min(i,3)).plot(self.eplist[:self.ep+1], ydata[:self.ep+1], frmt, label=label_+label)
            plt.xlabel('episodes')
            plt.legend()
            i+=1
        
        # if there is any visualisation required then we need to care for special cases    
        if self.plotV or self.plotE or self.plotT or self.plotR:
            figsizes = list(zip(plt.gcf().get_size_inches(), self.env.figsize0))
            figsize  = [max(figsizes[0]), min(figsizes[1]) if self.plotV or self.plotE else figsizes[1][0]]
            plt.gcf().set_size_inches(figsize[0], figsize[1])
            clear_output(wait=True)
            if not plot_exp: plt.show()


    def plot_V(self, ep=0):
        
        self.env.ax0 = plt.subplot(1,3,1) # to add this axis next to a another axis to save some spaces
#         plt.gcf().set_size_inches(16, 2)
        
        # get letter as state names if no more than alphabet else just give them numbers
        letters = self.env.letters_list()[1:-1] if self.env.nS<27 else list(range(self.env.nS-2))
        
        # plot the estimated values against the optimal values
        plt.plot(letters, self.V_()[1:-1], '.-', label='V episode=%d'%(ep)) # useful for randwalk
        plt.plot(letters, self.Vstar[1:-1],'.-k')
        
        # set up the figure
        plt.xlabel('State', fontsize=10)
        plt.legend()
        plt.title('Estimated value for %d non-terminal states'%(self.env.nS-2), fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        

# --------------------------------------- Multi-step  MRP-------------------------------------------

'''

    We develop our MRP class to accommodate waiting for n-1 steps before obtaining the $G_{t:t+n}$. 
    This is necessary to be able to implement algorithms that incorporate n-steps rewards instead 
    of the latest rewards. In each step, we must also create a G function to obtain the $G_{t:t+n}$.
    Finally, we would need to alter our stopping criteria to wait for extra n-1 steps at the end 
    to ensure we update the latest n-1 state values since we are always lagging n-1 steps during 
    the episode.
    all other *prediction algorithms* must inherit this class
'''
class MRP(MRP):
    def __init__(self, n=1, **kw):
        super().__init__(**kw)
        self.n = n
    #----------------------------------- steps as per the algorithm style --------------------------
    def stop_ep(self, done):
        return self.stop_(done) or (self.t+1 >= self.max_t-1)

    def stop_(self,done):
        if done:
            self.skipstep +=1                     # holds the count for how many steps after ep is finished (will reach n-1)
            if self.skipstep == self.n: 
                self.t = self.t+1 - self.skipstep # returns t to its original actual count of number of steps.  it is executed before self.t+=1 in interact and hence +1 is necessary
                self.skipstep = 0
                return True
            return False
        self.skipstep = 0
        return False
    #----------------------------------- returns --------------------------------------------------
    def G(self, τ1, τn):    # n-steps return, called during an episode
        #if self.γ==1: return self.r[τ1:τn+1].sum() # this saves computation when no dsicount is applied
        Gn = 0
        for t in range(τn, τ1-1, -1): # yields τn-τ1= (τ+n)-(τ+1)= n-1 setps
            Gn = self.γ*Gn + self.r[t]
        return Gn 

# =======================================Base class for control===============================================
'''
    all other *value control algorithms* must inherit this class
'''
def MDP(MRP=MRP):
    class MDP(MRP):
        def __init__(self, env=grid(), commit_ep=0, ε=.1, εmin=0.01, dε=1, εT=0, q0=0, Tstar=0, **kw): 

            super().__init__(env=env, **kw)
            # set up hyper parameters
            self.ε = ε 
            self.ε0 = ε  # store initial 
            self.dε = dε # for exp decay
            self.εT = εT # for lin decay
            self.εmin = εmin
            
            # override the policy to εgreedy to make control possible
            self.policy = self.εgreedy

            # initial Q values
            self.q0 = q0

            # which episode to commit changes
            self.commit_ep = commit_ep
            
            # number of steps for optimal policy
            self.Tstar = Tstar
            
        # set up the Q table
        def init_(self):
            super().init_() # initialises V
            self.Q = np.ones((self.env.nS, self.env.nA))*self.q0
        
        #------------------------------------- add some more policies types 易-------------------------------
        # useful for inheritance, gives us a vector of actions values
        def Q_(self, s=None, a=None):
            return self.Q[s] if s is not None else self.Q

        # directly calculates V as a π[s] policy expectation of Q[s] 
        def V_from_Q(self, s=None):
            return self.Q_(s)@self.π(s)
            
        # returns a pure greedy action, **not to be used in learning**
        def greedy_(self, s):
            return np.argmax(self.Q_(s))

        # greedy stochastic MaxQ
        def greedy(self, s): 
            self.isamax = True
            # instead of returning np.argmax(Q[s]) get all max actions and return one of the max actions randomly
            Qs = self.Q_(s)
            # print(s)
            # print(Qs)
            if Qs.shape[0]==1: raise ValueError('something might be wrong number of actions ==1')
            return choices(np.where(Qs==Qs.max())[0])[0] # more efficient than choice
            #return choice(np.where(Qs==Qs.max())[0])
        
        # returns a greedy action most of the time
        def εgreedy(self, s):
            # there is pr=ε/nA that a max action is chosen but is not considered max, we ignored it in favour of efficiency
            self.isamax = False 
            if self.dε < 1: self.ε = max(self.εmin, self.ε*self.dε)              # exponential decay
            if self.εT > 0: self.ε = max(self.εmin, self.ε0 - self.t_ / self.εT) # linear      decay
            
            return self.greedy(s) if rand() > self.ε else randint(0, self.env.nA)
    
        # returns the policy probabilities (of selecting a specific action)
        def π(self, sn,  a=None):
            ε, nA, Qsn = self.ε, self.env.nA, self.Q_(sn)
            π_ = Qsn*0 + ε/nA
            π_[Qsn.argmax()] += 1-ε
            return π_ if a is None else π_[a]

        # returns whether the current policy is optimal by checking if agent can reach the goal in self.Tstar
        def πisoptimal(self):
            s = self.env.reset()
            done = False
            for t in range(self.Tstar):
                s,_, done,_ = self.env.step(self.greedy_(s))
            return done

        #---------------------------------------visualise ✍️----------------------------------------
        # override the render function
        def render(self, rn=None, label='', **kw):
            if rn is None: rn=self.rn
            param = {'Q':self.Q_()} if 'Q' in self.underhood else {} # 'maxQ' or 'Q'
            self.env.render(**param, 
                            label=label+' reward=%d, t=%d, ep=%d'%(rn, self.t+1, self.ep+1), 
                            underhood=self.underhood, **kw)
    return MDP

# =============================Basic policy gradient 易 class for control===================================

'''
    all other *policy gradient control algorithms* must inherit this class

'''
def PG(MDP=MDP(MRP)):
    class PG(MDP):
        def __init__(self, τ=1, τmin=.1, dτ=1, Tτ=0, **kw):
            super().__init__(**kw)
            # set up hyper parameters
            self.τ = τ
            self.τ0 = τ
            self.dτ = dτ
            self.Tτ = Tτ
            self.τmin = τmin

            # softmax is the default policy selection procedure for Policy Gradient methods
            self.policy = self.τsoftmax

        #----------------------------------- add some more policies types 易-------------------------------
        # returns a softmax action
        def τsoftmax(self, s):
            Qs = self.Q_(s)
            
            if self.dτ < 1: self.τ = max(self.τmin, self.τ*self.dτ)              # exponential decay
            if self.Tτ > 0: self.τ = max(self.τmin, self.τ0 - self.t_ / self.Tτ) # linear      decay
                
            exp = np.exp(Qs/self.τ)
            maxAs = np.where(Qs==Qs.max())[0]
            #a = choice(self.env.nA, 1, p=exp/exp.sum())[0]
            a = choices(range(self.env.nA), weights=exp/exp.sum(), k=1)[0]
            self.isamax = a in maxAs
            return a

        # overriding π() in parent class MDP: 
        # in MDP π() returns probabilities according to a εgreedy,
        # in PG  π() returns probabilities accroding to a τsoftmax, while
        def π(self, s, a=None):
            Qs = self.Q_(s)
            exp = np.exp(Qs/self.τ)
            return exp/exp.sum() if a is None else (exp/exp.sum())[a]
        
    return PG

# =======================handy quick setting depending on the problem(prediction or control)==================
def demo(what='V'):
    switch = {
        'V':    {'plotE':True, 'plotV':True, 'animate':True},                    # suitable for prediction
        'T':    {'plotT':True, 'visual':True, 'underhood':'maxQ'},               # suitable for control
        'R':    {'plotR':True, 'visual':True, 'underhood':'maxQ'},               # suitable for control
        'TR':   {'plotT':True, 'plotR':True, 'visual':True,'underhood':'maxQ'},  # suitable for control
        'Game': {'plotT':True, 'plotR':True, 'visual':True, 'animate':True}      # suitable for games
    }
    return switch.get(what,{})
def demoV(): return demo('V')
def demoT(): return demo('T')
def demoQ(): return demo('T')# alias
def demoR(): return demo('R')
def demoTR(): return demo('TR')
def demoGame(): return demo('Game')

# ======================================================================================
'''
    offline here in the sense of end-of-episode learning, 
    not a pure offline where there is no inbetween episodes interaction
'''
# ------------------  offline Monte Carlo value function prediction learning -----------------------
class MC(MRP):
    def init(self):
        self.store = True
       
    def offline(self):
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            rn = self.r[t+1]
            
            Gt = self.γ*Gt + rn
            self.V[s] += self.α*(Gt - self.V[s])

# -------------------  offline Monte Carlo value function control learning 六‍ -----------------------
class MCC(MDP()):
    def init(self):
        self.store = True
        
    def offline(self):  
        # obtain the return for the latest episode
        Gt = 0
        for t in range(self.t, -1, -1):
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]

            Gt = self.γ*Gt + rn
            self.Q[s,a] += self.α*(Gt - self.Q[s,a])

# -------------------  offline, REINFORCE: MC for policy gradient 易 control methdos ----------------
class REINFORCE(PG()):
    def init(self):
        self.store = True
    
    def offline(self):
        π, γ, α, τ = self.π, self.γ, self.α, self.τ
        # obtain the return for the latest episode
        Gt = 0
        γt = γ**self.t                  # efficient way to calculate powers of γ backwards
        for t in range(self.t, -1, -1): # reversed to make it easier to calculate Gt
            s = self.s[t]
            a = self.a[t]
            rn = self.r[t+1]
            
            Gt = γ*Gt + rn
            δ = Gt - self.V[s]
            
            self.V[s]   += α*δ
            self.Q[s,a] += α*δ*(1 - π(s,a))*γt/τ
            γt /= γ


# --------------------  online Temporal Difference: value prediction learning ------------------------
class TD(MRP):  
    def online(self, s, rn,sn, done, *args): 
        self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])

# --------------------  offline Temporal Difference(TD): value prediction learning ----------------------
class TDf(MRP):
    def init(self):
        self.store = True
  
    def offline(self):
        #for t in range(self.t, -1, -1):
        for t in range(self.t+1):
            s = self.s[t]
            sn = self.s[t+1]
            rn = self.r[t+1]
            done = self.done[t+1]
            
            self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn]- self.V[s])

# --------------------  online multi-step TD: value prediction learning ---------------------------------
class TDn(MRP):
    def init(self):
        self.store = True # there is a way to save storage by using t%(self.n+1) but we left it for clarity
       
    def online(self,*args):
        τ = self.t - (self.n-1);  n = self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τn = τ+n ; τn = min(τn, self.t+1 - self.skipstep)
        τ1 = τ+1
        
        sτ = self.s[τ ]
        sn = self.s[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.V[sτ] += self.α*(self.G(τ1,τn) + (1- done)*self.γ**n *self.V[sn] - self.V[sτ])
    
# --------------------  offline multi-step TD: value prediction learning ---------------------------
class TDnf(MRP):
    def init(self):
        self.store = True # must store because it is offline
     
    def offline(self):
        n=self.n        
        for t in range(self.t+n): # T+n to reach T+n-1
            τ  = t - (n-1)
            if τ<0: continue
        
            # we take the min so that we do not exceed the episode limit (last step+1)
            τ1 = τ+1
            τn = τ+n ; τn=min(τn, self.t+1)
            
            sτ = self.s[τ ]
            sn = self.s[τn]
            done = self.done[τn]
            
            # n steps τ+1,..., τ+n inclusive of both ends
            self.V[sτ] += self.α*(self.G(τ1,τn)+ (1- done)*self.γ**n *self.V[sn] - self.V[sτ])

# --------------------  online Sarsa: value control learning -----------------------------------------
class Sarsa(MDP()):
    def init(self): #α=.8
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    def online(self, s, rn,sn, done, a,an):
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.Q[sn,an] - self.Q[s,a])

# --------------------  online multi-step Sarsa: value control learning -------------------------------
class Sarsan(MDP()):
    def init(self):
        self.store = True        # although online but we need to access *some* of earlier steps,
        self.step = self.step_an # for Sarsa we want to decide the next action in time step t
    
    # -----------------------------  online learning ----------------------    
    def online(self,*args):
        τ = self.t - (self.n-1);  n=self.n
        if τ<0: return
        
        # we take the min so that we do not exceed the episode limit (last step+1)
        τ1 = τ+1
        τn = τ+n ; τn=min(τn, self.t+1 - self.skipstep)
        
        sτ = self.s[τ];  aτ = self.a[τ]
        sn = self.s[τn]; an = self.a[τn]
        done = self.done[τn]
        
        # n steps τ+1,..., τ+n inclusive of both ends
        self.Q[sτ,aτ] += self.α*(self.G(τ1,τn) + (1- done)*self.γ**n *self.Q[sn,an] - self.Q[sτ,aτ])

# --------------------  online Q-learning: value control learning ------------------------------------
class Qlearn(MDP()):
    def online(self, s, rn,sn, done, a,_):
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.Q[sn].max() - self.Q[s,a])

# --------------------  online Expected Sarsa: value control learning --------------------------------
class XSarsa(MDP()):
    def online(self, s, rn,sn, done, a,_):      
        # obtain the ε-greedy policy probabilities, 
        # then obtain the expecation via a dot product for efficiency
        π = self.π(sn)
        v = self.Q[sn].dot(π)
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*v - self.Q[s,a])

# --------------------  online double Q-learning: value control learning -------------------------------
class DQlearn(MDP()):
    def init(self):
        self.Q1 = self.Q
        self.Q2 = self.Q.copy()
        
    # we need to override the way we calculate the aciton-value function in our εgreedy policy
    def Q_(self, s=None, a=None):
            return self.Q1[s] + self.Q2[s] if s is not None else self.Q1 + self.Q2

    def online(self, s, rn,sn, done, a,_): 
        p = np.random.binomial(1, p=0.5)
        if p:    self.Q1[s,a] += self.α*(rn + (1- done)*self.γ*self.Q2[sn].max() - self.Q1[s,a])
        else:    self.Q2[s,a] += self.α*(rn + (1- done)*self.γ*self.Q1[sn].max() - self.Q2[s,a])

# --------------------  online Actor-Critic: policy gradient 易 control learning ------------------------
class Actor_Critic(PG()):
    def step0(self):
        self.γt = 1 # powers of γ, must be reset at the start of each episode
    
    def online(self, s, rn,sn, done, a,an): 
        π, γ, γt, α, τ, t = self.π, self.γ, self.γt, self.α, self.τ, self.t
        δ = (1- done)*γ*self.V[sn] + rn - self.V[s]  # TD error is based on the critic estimate

        self.V[s]   += α*δ                          # critic
        self.Q[s,a] += α*δ*(1- π(s,a))*γt/τ         # actor
        self.γt *= γ


# ===================adding a few functions that will serve us well for comparing famous algorithms=====================
def TD_MC_randwalk(env=randwalk(), alg1=TDf, alg2=MC):
    plt.xlim(0, 100)
    plt.ylim(0, .25)
    plt.title('Empirical RMS error, averaged over states')
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=alg1(env=env, α=α, v0=.5), runs=100, plotE=True).interact(label='TD α= %.2f'%α, frmt='-')

    for α in [.01, .02, .03, .04]:
        MCs = Runs(algorithm=alg2(env=env, α=α, v0=.5), runs=100, plotE=True).interact(label='MC α= %.2f'%α, frmt='--')

def example_6_2(**kw): return TD_MC_randwalk(**kw)

# ----------------------------------------------------------------------------------------------------------------------
def figure_6_2():
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(0,100)
    plt.ylim(0, .25)
    plt.title('Batch Training')

    α=.001
    TDB = Runs(algorithm=TD_batch(v0=-1, α=α, episodes=100), runs=100, plotE=True).interact(label= 'Batch TD, α= %.3f'%α)
    MCB = Runs(algorithm=MC_batch(v0=-1, α=α, episodes=100), runs=100, plotE=True).interact(label='Batch MC, α= %.3f'%α)
# ----------------------------------------------------------------------------------------------------------------------
def Sarsa_windy():
    return Sarsa(env=windy(reward='reward1'), α=.5, seed=1, **demoQ(), episodes=170).interact(label='TD on Windy')
    
example_6_5 = Sarsa_windy

# ----------------------------------------------------------------------------------------------------------------------
def Sarsa_Qlearn_cliffwalk(runs=200, α=.5, env=cliffwalk(), alg1=Sarsa, alg2=Qlearn):
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    
    plt.yticks([-100, -75, -50, -25])
    plt.ylim(-100, -10)
    
    SarsaCliff = Runs(algorithm=alg1(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Sarsa')
    QlearnCliff = Runs(algorithm=alg2(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Q-learning')
    return SarsaCliff, QlearnCliff

def example_6_6(**kw): 
    return Sarsa_Qlearn_cliffwalk(**kw)

# ----------------------------------------------------------------------------------------------------------------------
def XSarsaDQlearnCliff(runs=300, α=.5):
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)    
    plt.yticks([-100, -75, -50, -25])
    plt.ylim(-100, -10)
    env = cliffwalk()

    XSarsaCliff = Runs(algorithm=XSarsa(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='XSarsa')
    DQlearnCliff = Runs(algorithm=DQlearn(env=env, α=α, episodes=500), runs=runs, plotR=True).interact(label='Double Q-learning')

    return XSarsaCliff, DQlearnCliff

# ----------------------------------------------------------------------------------------------------------------------
def compareonMaze(runs=100, α=.5):
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    env=Grid(gridsize=[10,20], style='maze', s0=80, reward='reward1') # this is bit bigger than the defualt maze
    env.render()
    
    SarsaMaze = Runs(algorithm=Sarsa(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Sarsa')
    XSarsaMaze = Runs(algorithm=XSarsa(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='XSarsa')
    
    QlearnMaze = Runs(algorithm=Qlearn(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Q-learning')
    DQlearnMaze = Runs(algorithm=DQlearn(env=env, α=α, episodes=30), runs=runs, plotT=True).interact(label='Double Q-learning')

    return SarsaMaze, XSarsaMaze, QlearnMaze, DQlearnMaze

# ----------------------------------------------------------------------------------------------------------------------
def figure_6_3(runs=10, Interim=True, Asymptotic=True, episodes=100,  label=''): #100
    #plt.ylim(-150, -10)
    plt.xlim(.1,1)
    plt.title('Interim and Asymptotic performance')
    αs = np.arange(.1,1.05,.05)

    algors = [ XSarsa,   Sarsa,   Qlearn]#,      DQlearn]
    labels = ['XSarsa', 'Sarsa', 'Qlearning']#, 'Double Q learning']
    frmts  = ['x',      '^',     's']#,         'd']
    
    env = cliffwalk()
    Interim_, Asymptotic_ = [], []
    # Interim perfromance......
    if Interim:
        for g, algo in enumerate(algors):
            compare = Compare(algorithm=algo(env=env, episodes=episodes), runs=runs, hyper={'α':αs},
                             plotR=True).compare(label=labels[g]+' Interim'+label, frmt=frmts[g]+'--')
            Interim_.append(compare)
    
    # Asymptotic perfromance......
    if Asymptotic:
        for g, algo in enumerate(algors):
            compare = Compare(algorithm=algo(env=env, episodes=episodes*10), runs=runs, hyper={'α':αs}, 
                             plotR=True).compare(label=labels[g]+' Asymptotic'+label, frmt=frmts[g]+'-')
            Asymptotic_.append(compare)
    
    plt.gcf().set_size_inches(10, 7)
    return Interim_, Asymptotic_
    

# ----------------------------------------------------------------------------------------------------------------------
def nstepTD_MC_randwalk(env=randwalk(), algorithm=TDn, alglabel='TD'):
    plt.xlim(0, 100)
    plt.ylim(0, .25)
    plt.title('Empirical RMS error, averaged over states')
    n=5
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=algorithm(env=env, n=1,α=α, v0=.5),  runs=100, plotE=True).interact(label='%s α= %.2f'%(alglabel,α), frmt='.-')
    
    for α in [.05, .1, .15]:
        TDαs = Runs(algorithm=algorithm(env=env,n=n,α=α, v0=.5),  runs=100, plotE=True).interact(label= '%s α= %.2f n=%d'%(alglabel,α,n), frmt='-')

    for α in [.01, .02, .03, .04]:
        MCs = Runs(algorithm=MC(env=env,α=α, v0=.5),  runs=100, plotE=True).interact(label='MC α= %.2f'%α, frmt='--')

# ----------------------------------------------------------------------------------------------------------------------
def nstepTD_MC_randwalk_αcompare(env=randwalk_(), algorithm=TDn, Vstar=None, runs=10, envlabel='19', 
                                 MCshow=True, alglabel='online TD'):
    
    steps0 = list(np.arange(.001,.01,.001))
    steps1 = list(np.arange(.011,.2,.025))
    steps2 = list(np.arange(.25,1.,.05))

    αs = np.round(steps0 +steps1 + steps2, 2)
    #αs = np.arange(0,1.05,.1) # quick testing
    
    plt.xlim(-.02, 1)
    plt.ylim(.24, .56)
    plt.title('n-steps %s RMS error averaged over %s states and first 10 episodes'%(alglabel,envlabel))
    for n in [2**_ for _ in range(10)]:
        Compare(algorithm=algorithm(env=env, v0=0, n=n, episodes=10, Vstar=Vstar), 
                              runs=runs, 
                              hyper={'α':αs}, 
                              plotE=True).compare(label='n=%d'%n)
    if MCshow:
        compare = Compare(algorithm=MC(env=env, v0=0, episodes=10), 
                                  runs=runs, 
                                  hyper={'α':αs}, 
                                  plotE=True).compare(label='MC ≡ TDn(n=$\\infty$)', frmt='-.')
# ----------------------------------------------------------------------------------------------------------------------
def figure_7_4(n=5,seed=16): 
    
    # draw the path(trace) that the agent took to reach the goal
    nsarsa = Sarsan(env=grid(), α=.4, seed=seed, episodes=1).interact()
    nsarsa.env.render(underhood='trace', subplot=131, animate=False, label='path of agent')

    # now draw the effect of learning to estimate the Q action-value function for n=1
    nsarsa = Sarsan(env=grid(), α=.4, seed=seed, episodes=1, underhood='maxQ').interact() 
    nsarsa.render(subplot=132, animate=False, label='action-value increassed by 1-steps Sarsa\n')
    
    #n=5 # try 10
    # now draw the effect of learning to estimate the Q action-value function for n=10
    nsarsa = Sarsan(env=grid(), n=n, α=.4, seed=seed, episodes=1, underhood='maxQ').interact()    
    nsarsa.render(subplot=133, animate=False, label='action-value increassed by %d-steps Sarsa\n'%n)
# ----------------------------------------------------------------------------------------------------------------------
