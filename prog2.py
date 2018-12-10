
# coding: utf-8

# In[1]:


import csv


# In[2]:


def more_general(h1,h2):
    more_general_parts = []
    for x, y in zip(h1,h2):
        mg = x =="?" or (x != "0" and (x == y or y =="0"))
        more_general_parts.append(mg)
    return all(more_general_parts)
    
l1 = [1,2,3]
l2 = [3,4,5]

list(zip(l1,l2))


# In[3]:


def fulfills(example, hypothesis):
    return more_general(hypothesis, example)

def min_generalizations(h,x):
    h_new = list(h)
    for i in range(len(h)):
        if not fulfills(x[i:i+1],h[i:i+1]):
            h_new[i] = "?" if h[i] != '0' else x[i]
    return [tuple(h_new)]


# In[4]:


min_generalizations(h=('0','0','sunny'),x=('rainy','windy','cloudy'))


# In[5]:


def min_specializations(h,domains,x):
    results = []
    for i in range(len(h)):
        if h[i] == "?":
            for val in domains[i]:
                if x[i] != val:
                    h_new =h[:i] + (val,) + h[i+1:]
                    results.append(h_new)
        elif h[i] != "0":
            h_new =h[:i] + ('0',) + h[i+1:]
            results.append(h_new)
    return results


# In[6]:


min_specializations(h=('?','x',),domains=[['a','b','c'],['x','y']],x=('b','x'))


# In[7]:


with open('train2.csv') as csvFile:
    examples = [tuple(line) for line in csv.reader(csvFile)]

examples


# In[8]:


l = [x for x in range(4)]


# In[9]:


def get_domains(examples):
    d = [set() for i in examples[0]]
    print(d)
    for x in examples:
        for i, xi in enumerate(x):
            d[i].add(xi)
        print(d)
    return [list(sorted(x)) for x in d]
get_domains(examples)


# In[10]:


def specialize_G(x,domains,G,S):
    G_prev = list(G)
    for g in G_prev:
        if g not in G:
            continue
        if fulfills(x,g):
            G.remove(g)
            Gminus = min_specializations(g,domains,x)
            G.update([h for h in Gminus if any([more_general(h,s) for s in S])])
            G.difference_update([h for h in G if any([more_general(g1,h) for g1 in G if h != g1])])
    return G


# In[11]:


def generalize_S(x,G,S):
    S_prev = list(S)
    for s in S_prev:
        if s not in S:
            continue
        if not fulfills(x,s):
            S.remove(s)
            Splus = min_generalizations(s,x)
            S.update([h for h in Splus if any([more_general(g,h) for g in G])])
            S.difference_update([h for h in S if any([more_general(h,h1) for h1 in S if h != h1])])
    return S


# In[12]:


def candidate_elimination(examples):
    domains = get_domains(examples)[:-1]
    
    G = set([("?",)*len(domains)])
    S = set([("0",)*len(domains)])
    i=0
    print("\n G[{0}]:".format(i),G)
    print("\n S[{0}]:".format(i),S)
    for xcx in examples:
        i=i+1
        x, cx = xcx[:-1], xcx[-1]
        if cx == "Y":
            G = {g for g in G if fulfills(x,g)}
            S = generalize_S(x,G,S)
        else:
            S = {s for s in S if not fulfills(x,s)}
            G = specialize_G(x,domains,G,S)
        print("\n G[{0}]:".format(i),G)
        print("\n S[{0}]:".format(i),S)
    return S,G


# In[13]:


def enumerateHypothesesBetween_s_g(s,g):
    hypotheses = []
    
    for i, constraint in enumerate(g):
        if constraint != s[i]:
            hypothesis = g[:]
            hypothesis[i]=s[i]
            hypotheses.append(hypothesis)
    return hypotheses
        


# In[14]:


def enumerateVersionSpace(S,G):
    hypotheses = []
    hypotheses += S
    hypotheses += G
    print("Initial Hypothesis ",hypotheses)
    s = hypotheses[0]
    for i in range(1,len(hypotheses)):
        inBetweenHypotheses = enumerateHypothesesBetween_s_g(list(s),list(hypothese[i]))
        hypotheses.extend(inBetweenHypotheses)
    print("Hypothesis with duplicates ",hypotheses)
    
    setH = set()
    for h in hypotheses:
        setH.add(tuple(h))
        
    ans=[list(x) for x in setH]
    print("Version Space: ",ans)
    


# In[15]:


S,G=candidate_elimination(examples)


# In[16]:


examples


# In[17]:


S


# In[18]:


G

