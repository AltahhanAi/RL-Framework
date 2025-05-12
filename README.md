This is a full framework for Reinforcement Learning that allows a developer to develop algorithms in a straightforward manner.
The developer needs to download the env and rl folders and then create a TD or Qlearn classes, one can do something like:

from rl.rl import *

# prediction algorithm example and application

```
class TD(MRP):

    # ----------------------------- 🌖 online learning ----------------------    
    
    def online(self, s, rn,sn, done, *args): 
        self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])


TDwalk = TD(episodes=100, v0=.5, **demoV()).interact(label='TD learning')
```




# control algorithm example and application
```
class Qlearn(MDP()):

    #--------------------------------------🌖 online learning --------------------------------------
    
    def online(self, s, rn,sn, done, a,_):
        self.Q[s,a] += self.α*(rn + (1- done)*self.γ*self.Q[sn].max() - self.Q[s,a])


qlearn = Qlearn(env=maze(), α=.5, episodes=50, seed=1, **demoR()).interact()
```
The framework also provides support for robotics and extensive examples that cover most of Sutton and Barto's RL book!
