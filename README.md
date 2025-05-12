This is a full framework for Reinforcement Learning that allows a developer to develop algorithms in straightforward manner.
The developer needs to download the env and rl folders and then to create a TD class, once can do something like:

from rl.rl import *

class TD(MRP):
    # ----------------------------- 🌖 online learning ----------------------    
    def online(self, s, rn,sn, done, *args): 
        self.V[s] += self.α*(rn + (1- done)*self.γ*self.V[sn] - self.V[s])


