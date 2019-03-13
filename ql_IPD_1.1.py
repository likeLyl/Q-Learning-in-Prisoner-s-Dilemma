import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#iterated prisoner's dilemma
#version 1.1
#opponent strategy: TFT, RANDOM, Q
#                  tit-for-tat,random,Q-learning
#state: the last myaction and opponent action.
#available action: cooperation and betray

#depend on the last state to make a decision (choose an action)


#depend on present time to create a true random seed.
#then, I can create a true random number
now_time = time.time() * 100000
time_1 = now_time // 100
time_seed = round(now_time % time_1)
np.random.seed(time_seed)
#end






class Q_learning():
    def __init__(self):
        #normal parameter
        self.ACTIONS = ['c', 'b']  # c:coopreate, b:betray
        self.STATE=['cc','cb','bc','bb']
        self.learning_rate = 0.1
        self.gamma = 0.6
        self.iteration = 5000
        #self.first=True
        self.T = 1000
        self.myAgent_action='c'

        self.reward = 0
        self.choose_c=0   #the number of cooperation
        self.E=[]
        self.ctimes=[]

        #opponent parameter
        self.opponent_action = 'c'
        self.opponent_reward=0
        self.opponent_choosec=0
        self.opponent_strategy='TFT'
        self.opponent_ctimes = []

    def Q_table_init(self):
        table=pd.DataFrame(np.zeros((len(self.STATE), len(self.ACTIONS))),index=self.STATE,columns=self.ACTIONS)
        return table


    def Q_value_update(self,q_table,state,action,opponent_action):
        if state=='init':
            s, rew = self.s_r(action, opponent_action)
            return s
        else:
            s,rew=self.s_r(action, opponent_action)    #according previous state, now action, observe r and now state

            q_pre=q_table.loc[state,action]

            q_value=rew+self.gamma*q_table.loc[s,:].max()
            q_table.loc[state,action] += self.learning_rate*(q_value-q_pre)
            #self.E_function(q_pre,q_value)
            return s

    def s_r(self,action,opponent_action):
        #tit-for-tat
        #random
        #Q-learning
        #if self.opponent_strategy != 'Q':           #if opponent strategy is Q learning,
        #    self.change_opp_action(state)           #the opponent action will
        #    opponent_action=self.opponent_action    #be updated outside of this “if”
        if opponent_action=='c':
            if action=='c':  #cc  3
                rew=3
                sta='cc'
                #self.choose_c+=1
            else:
                rew=5     #bc  5
                sta='bc'
        else:
            if action=='c':  #cb  0
                rew=0
                sta='cb'
                #self.choose_c += 1
            else:
                rew=1     #bb   1
                sta='bb'
        return sta,rew


    def choose_action(self,state,q_table):   #policy
        if state=='init':
            return np.random.choice(self.ACTIONS)
        else:
            qa=q_table.loc[state,'c']
            qb=q_table.loc[state,'b']
            #print(qa)
            #ps_c=np.exp(qa/self.T)/(np.exp(qa/self.T)+np.exp(qb/self.T))
            ps_b=np.exp(qb/self.T)/(np.exp(qa/self.T)+np.exp(qb/self.T))

            #print('------')
            #print('probability of cooperation: ',ps_c)
            #print(ps_b)
            #print('------')

            if np.random.random()>ps_b:
                return 'c'
            elif np.random.random()<ps_b:
                return 'b'
            else:
                #print("random:")
                return np.random.choice(self.ACTIONS)

    def change_opp_action(self,state):

        strategy=self.opponent_strategy

        if(strategy=='TFT'):
            if state == 'cc':
                self.opponent_action='c'
            if state == 'cb':
                self.opponent_action = 'c'
            if state == 'bc':
                self.opponent_action = 'b'
            if state == 'bb':
                self.opponent_action = 'b'
            if state == 'init':
                self.opponent_action='c'
        if(strategy=='RANDOM'):
            self.opponent_action=np.random.choice(self.ACTIONS)
        #if(strategy=='Q'):


    def sum_reward(self,state):
        if state=='cc':
            self.reward += 3
            self.opponent_reward+=3
        if state == 'cb':
            self.reward += 0
            self.opponent_reward +=5
        if state == 'bc':
            self.reward += 5
            self.opponent_reward +=0
        if state == 'bb':
            self.reward += 1
            self.opponent_reward +=1


    def sum_choose(self,myagent_action,op_action):
        if myagent_action=='c':
            self.choose_c+=1
            self.ctimes.append(self.choose_c)
        else:
            self.ctimes.append(self.choose_c)
        if op_action=='c':
            self.opponent_choosec+=1
            self.opponent_ctimes.append(self.opponent_choosec)
        else:
            self.opponent_ctimes.append(self.opponent_choosec)

    def E_function(self,q_pre,q_value):
        self.E.append(np.square(q_value-q_pre))


    def train(self,opponent_strategy='TFT',iteration=5000,T=1000,gamma=0.9,learning_rate=0.01):
        q_table = self.Q_table_init()
        q_table_opponent = self.Q_table_init()
        self.opponent_strategy=opponent_strategy
        self.iteration=iteration
        self.T=T
        self.gamma=gamma
        self.learning_rate=learning_rate

        S='init'

        opponent_State='init'
        opponent_Action=self.opponent_action

        #opponent agent choose action
        if self.opponent_strategy=='Q':
            opponent_Action=self.choose_action(opponent_State,q_table_opponent)
        else:

            self.change_opp_action(opponent_State)
            opponent_Action=self.opponent_action
            #self.opponent_action=opponent_Action

        #my agent choose action
        A=self.choose_action(S,q_table)
        self.sum_choose(A,opponent_Action)

        #opponent agent update state and q_table
        if self.opponent_strategy=='Q':
            opponent_State = self.Q_value_update(q_table_opponent, opponent_State, opponent_Action,A)

        #my agent update state and
        S=self.Q_value_update(q_table,S,A,opponent_Action)

        self.sum_reward(S)

        for episode in range(self.iteration):
            #self.T in softmax(choose_action())
            if self.T>1:
                self.T -=1

            A=self.choose_action(S, q_table)

            #self.myAgent_action = A
            if self.opponent_strategy == 'Q':
                opponent_Action = self.choose_action(opponent_State, q_table_opponent)
                #self.opponent_action = opponent_Action
                opponent_State = self.Q_value_update(q_table_opponent, opponent_State, opponent_Action,A)
            else:
                self.change_opp_action(S)
                opponent_Action = self.opponent_action

            S = self.Q_value_update(q_table, S, A,opponent_Action)
            self.sum_reward(S)
            self.sum_choose(A, opponent_Action)
        print('myagent: table\n',q_table)
        if self.opponent_strategy=='Q':
            print('opponent agent table:\n',q_table_opponent)
        print('opponent strategy:',self.opponent_strategy)
        print('myagent reward:',self.reward)
        print('opponent reward:',self.opponent_reward)
        print('myagent co:', self.choose_c)
        print('opponent co:', self.opponent_choosec)

        #plot the number of choose co figure
        x=[]
        for i in range(self.iteration+1):
            x.append(i)
        plt.xlim(0,self.iteration)
        plt.ylim(0, 10000)
        plt.figure(1)
        plt.scatter(x, self.ctimes, s=1,c='r',marker='.', label='myagent choose co')
        plt.scatter(x, self.opponent_ctimes,s=1, c='b',marker='.', label='opponent choose co')
        plt.xlabel('Iterations')
        plt.ylabel('the number of co')
        plt.legend(loc='upper left')

        plt.show()

if __name__ == '__main__':
    ql=Q_learning()
    ql.train(opponent_strategy='TFT',iteration=10000,T=1000,learning_rate=0.1)
