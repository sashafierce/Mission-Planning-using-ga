import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
from random import shuffle
from numpy import prod
import sys
import copy

random.seed(7)
w1 = 1 # weight to normalize the travelling time in cost function
w2 = 1 # weight to normalize the waiting time in cost function
w = 1 # hyperparameter for greedy heuristic
speed = 13.6 # pixels/second speed
# divide by max entry time 
no_of_sites = 31
tasks = 3 # sample mission.docs
point_of_failure = 1
start = 3  # robita entrance
sub_tasks = 4
crossover_rate = 0.9
prob_distribution = []
for i in range(no_of_sites):
    prob_distribution.append(random.random())

wait_distribution = []
for i in range(no_of_sites):
    wait_distribution.append(random.uniform(0,1))

#  For sample missions 1 and 2 and 3
# Task 1: User A wants his project report to be signed by any 2 panelists out of 
#  5, then to be submitted to either supervisor or office. 
#    A then (2 of # B, C, D) then (1 of P1 or S) 
# Task 2: User B wants to get his room keys which may be with any of this friend
#    (A, C or D). B then 1 of (A, C, D) then B
# Task 3: User C wants to get a stipend form printout from any of the machines 
#    X1 or X2 (either may be mal-functional) and after signing to be submitted to 
#    either of the supervisor for signature and finally given to office. 
#    c then 1 of (X1 or X2) then (P1 or P2) then S. 
# A = 9 then ( any 2 of (21, 28, 29) ) then (1 or 4) 
# B = (21, 22, 25) then (6)
# C = 21 then (7 or 8) then (1 or 2) then (4)

q = 100 # run q times for same distribution but random ground truth
while q > 0 :

    missionSites = []   
    # hardcoded mission specification  
    k = [[1, 2, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1]]
    m = [[1, 3, 2, 0],
            [3, 1, 0, 0],
            [1, 2, 2, 1]]
    siteIndex = [ [ [9], [21,28,29], [1,4], [] ],
                    [ [21,22,25], [6], [], [] ], 
                    [ [21], [7,8], [2 , 1], [4] ] ]
    k_copy = k[:]
    m_copy = m[:]

    class Site:
        def __init__(self, i, x, y, wtime, name, prob):
            self.index = i
            self.x = x
            self.y = y
            self.title = name
            self.wait_time = wtime
            self.avail_probability = prob
            self.mission = []
            self.group = []
            self.tuple = ()
        
        def cost(self, site):
            
            dis = cost_matrix[self.index-1][site.index-1]
            time = w1 * dis / speed + w2 * site.wait_time
            prob_fail = 1 - site.avail_probability
            cost = w * time + prob_fail 
            return cost
        
        def distance(self, site):
            dis = cost_matrix[self.index-1][site.index-1]
            time = w1 * dis / speed + w2 * site.wait_time
            cost =  time
            return cost
              
        def __repr__(self):
            return "(" + str(self.index) + "," + str(self.x) + "," + str(self.y) + "," + str(self.title) + "," + str(self.wait_time) + "," + str(self.avail_probability) +  "," + str(self.mission) + "," + str(self.group) + "," +  str(self.tuple) +")"
            
    class Fitness:
        def __init__(self, route):
            self.route = route
            self.cost = 0
            self.fitness= 0.0
        
        def routeDistance(self):
            if self.cost == 0:
                pathCost = 0
                fromSite = start_site
                w = 1
                toSite = self.route[0]
                pathCost += fromSite.cost(toSite)
                prob = 1

                for i in range(tasks):
                    task_prob = 1
                    for j in range(sub_tasks):
                        sub_prob = 1
                        for site in self.route:
                            if j in site.group and i in site.mission :
                                if k[i][j] == m[i][j]:  # ANDS
                                    sub_prob =  sub_prob * (1 - (1-site.avail_probability) )
                                else :
                                    sub_prob =  sub_prob * (1-site.avail_probability )
                                #print "site ", site
                        if not k[i][j] == m[i][j]:
                            sub_prob = 1 - sub_prob 
                        task_prob = task_prob * sub_prob   
                    prob = prob * task_prob 
                   
                for i in range(0, len(self.route)):
                    fromSite = self.route[i]
                    toSite = None

                    if i + 1 < len(self.route):
                        toSite = self.route[i + 1]
                    else:
                        return pathCost + prob
                    pathCost += fromSite.distance(toSite)
                
                self.cost = pathCost + prob
            return self.cost
        
        def routeFitness(self):
            if self.fitness == 0:
                self.fitness = 1 / float(self.routeDistance()) # maximise fitness, minimise cost
                #print "fitness = ", self.fitness
            return self.fitness
        
        def routeLength(self):
            if self.cost == 0:
                pathCost = 0
                fromSite = start_site
                w = 1
                toSite = self.route[0]
                pathCost += fromSite.cost(toSite)
                
                for i in range(0, len(self.route)):
                    fromSite = self.route[i]
                    toSite = None
                    if i + 1 < len(self.route):
                        toSite = self.route[i + 1]
                    else:
                        return pathCost
                    pathCost += fromSite.distance(toSite)
                self.cost = pathCost
            return self.cost
            
    def createRoute(missionSites):
        route = []

        for i in range(tasks):        
            for j in range(sub_tasks):

                size = random.randint(k[i][j], m[i][j])
                l = random.sample(missionSites[i][j], size)
                if len(l):
                    for item in l:
                        item.tuple = (i,j)
                        route.append(item)
     
        #route = random.sample(route, len(route) )	
        return route 

    def initialPopulation(popSize, missionSites):
        population = []

        for i in range(0, popSize):
            population.append(createRoute(missionSites))
        return population

    def rankRoutes(population):
        fitnessResults = {}

        for i in range(0,len(population)):
            fitnessResults[i] = Fitness(population[i]).routeFitness()
        result =  sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
        #print result
        return result

    def selection(popRanked, eliteSize):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
        for i in range(0, eliteSize):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - eliteSize):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i,3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults
        
    def matingPool(population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool


    def breed(parent1, parent2):
        child = []
        pool = []
        for task in range(tasks):
            sub_pool = []
            for group in range(sub_tasks):
                sub_pool.append( list( set( [site for site in parent1 if group == site.tuple[1] if task == site.tuple[0]] + [site for site in parent2 if group == site.tuple[1] if task == site.tuple[0] ] ) ) )
            pool.append(sub_pool)
                
        route = []  
        for i in range(tasks):
         #  sub_route = []
            for j in range(sub_tasks):
                size = min(random.randint(k[i][j], m[i][j]),len(pool[i][j]) )
                l = random.sample(pool[i][j], size) 
                if len(l):
                    for item in l:
                        route.append(item)
     
        # aligning according to greedy heuristic
        prev = start
        c = [[0]*sub_tasks]*tasks  # counter for dependent set elements, to ensure valid route
        
        while len(route) > 0:

            #ind = site.index
            sorted_cost_list = [ (total_cost_matrix[prev+1][site.index+1],site) for site in route ] 
            sorted_cost_list.sort()
            l = len(sorted_cost_list)
            i = 0

            while i < l:
                site = sorted_cost_list[i][1]
                i+=1
                groupi =  site.tuple[1]
                missioni = site.tuple[0]
                if groupi > 0 and (c[missioni][groupi-1] < k[missioni][groupi-1]):
                    continue
                else:
                    break
                
            c[site.tuple[0]][site.tuple[1]] +=1
            child.append(site)
            route.remove(site)
            prev = site.index
 
        return child

    def breedPopulation(matingpool, eliteSize):
        children = []
        length = len(matingpool) - eliteSize
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0,eliteSize):
            children.append(matingpool[i])
        
        for i in range(0, length):
            child = breed(pool[i], pool[len(matingpool)-i-1])
            children.append(child)
        return children

    # swap mutation with ensuring validity of site sequence
    def mutate(individual, mutationRate):
        for swapped in range(0,len(individual)):
            if(random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))
                maxi = max(swapped,swapWith)
                mini = min(swapped,swapWith)
                swapped = maxi
                swapWith = mini
                site1 = individual[swapped] # higher index site
                site2 = individual[swapWith]

                # don't swap if group index is greater for higher index site in case they belong to same mission 
                f = 0            
                if site1.tuple[0]  ==  site2.tuple[0] :   # match mission   
                    if site1.tuple[1] == site2.tuple[1] :   # match subtask  
                        f = 1
                if f == 0:
                    continue
                individual[swapped] = site2
                individual[swapWith] = site1
        return individual

    # function to run mutation over entire population
    def mutatePopulation(population, mutationRate, eliteSize):
        mutatedPop = []
        length = len(population) - eliteSize
        for i in range(0, eliteSize):
            mutatedPop.append(population[i])
        # mutation only for individuals except for elite
        for ind in range(eliteSize, len(population)):
            mutatedInd = mutate(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        return mutatedPop

    # all steps together to create the next generation
    def nextGeneration(currentGen, eliteSize, mutationRate):
        popRanked = rankRoutes(currentGen)
        selectionResults = selection(popRanked, eliteSize)
        matingpool = matingPool(currentGen, selectionResults)
        children = breedPopulation(matingpool, eliteSize)
        nextGeneration = mutatePopulation(children, mutationRate, eliteSize)
        return nextGeneration

    # genetic algorithm with variable length genomes
    def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
        pop = initialPopulation(popSize, population)
        print("Initial fitness: " + str(float(rankRoutes(pop)[0][1])))
        print("Initial cost: " + str(1 / float(rankRoutes(pop)[0][1])))
        cost_hist = []
        for i in range(0, generations):
            pop = nextGeneration(pop, eliteSize, mutationRate)
            cost_hist.append(1 / float(rankRoutes(pop)[0][1]))
        
        print("Final cost: " + str(1 / rankRoutes(pop)[0][1]))
        bestRouteIndex = rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        # only show convergence plot for first iteration of GA
        if itr == 0:
            plt.figure("cost vs iterations") 
            plt.plot(np.linspace(0,len(cost_hist),len(cost_hist)),np.array(cost_hist),label='cost vs iterations')  
            plt.xlabel('iteration')
            plt.ylabel('cost') 
            plt.show()
        
        return bestRoute

    f = open ( 'transition costs.txt' , 'r')

    # Read the cost matrix to store the opimised path distance between all sites
    cost_matrix = np.array([[float(num) for num in line.split('\t')] for line in f ])
    max_cost = sorted(list(set(cost_matrix.flatten().tolist())))[-2]
    w1 = float(speed) / max_cost  # normalize

    # populate siteList, contains all sites in map
    siteList = []
    with open('coord.txt', 'r') as file:
        i = 0
        
        for line in file :
            site = line.split()
            wtime = wait_distribution[i]
            prob = prob_distribution[i]  
            site = Site(i, float(site[1]), float(site[2]), wtime, site[0], prob)
            if i == start:
                start_site = site
            siteList.append(site)
            i+=1

    # populate missionSites list, contains sites as per mission description
    i = 0    
    for taski in siteIndex:
            task = []
            j = 0
            for subtaski in taski:     
                    subtask = []
                    for sitei in subtaski:
                            site = siteList[sitei]
                            site.group.append(j)
                            site.mission.append(i)
                            s = copy.copy(site)
                            s.tuple = (taski, subtaski)
                            subtask.append(s)
                    task.append(subtask)
                    j = j + 1 
            missionSites.append(task) 
            i = i + 1

    n = len(cost_matrix)
    total_cost_matrix = [[0 for x in range(n)] for y in range(n)] 
    for r in range(n):
        for c in range(n):
            total_cost_matrix[r][c] = siteList[r].cost(siteList[c])

    home = start_site
    siteList.remove(start_site)
    org_siteList = siteList[:]

    # generate ground truth randomly(uniform distribution)
    ground_truth = []
    for site in siteList:
        r = random.random()
        if r >= site.avail_probability:
            ground_truth.append(1)
        else:
            ground_truth.append(0)
    
    ground_truth[3] = 1 # gate, always available
    ground_truth[5] = 1
    fail = 0
    itr = 0
    final = []   # contains the total path including failed site visits
    avail = []   # stores availability of each visited site
    fails = [[0]*sub_tasks]*tasks  # counter for unavailable sites in each set, to abort dependent subtasks
    cost = 0
    cnt = 9 # no. of non-emtpy subtasks

    # run genetic algorithm, simulate using ground truth, until mission is complete
    while( cnt > 0) :
        print "\nRunning GA, itr = ", itr
        itr+=1
        route = geneticAlgorithm(population=missionSites, popSize=80, eliteSize=8, mutationRate=0.2, generations=80)
        l = len(route)
        final.append(route[0])
        
        group = route[0].tuple[1]
        mission = route[0].tuple[0]
        m[mission][group] -= 1        
        print "\nnext site ", route[0]
        if ground_truth[route[0].index]:
            avail.append("available")
            if k[mission][group]:
                k[mission][group] -= 1
                if k[mission][group] == 0:
                    cnt -= 1
            print "available"
        else:
            fail += 1
            avail.append("not available")
            print "not"
        # below is to handle scenerio when k = m initially for a task i.e all sites 
	# need to be visited but some of them are not available
        if k[mission][group] > m[mission][group]:
            k[mission][group] = int(m[mission][group])
            if k[mission][group] == 0:
                cnt -= 1
            
        start = route[0].index
        start_site = route[0]    
        missionSites[mission][group].remove(route[0])

    start_site = home # reset start_site to calc cost
    ind = 0
    for site in final:
	print site.title, "(",site.index,")", " ", site.tuple,")", " ", avail[ind],")", " ", site.avail_probability,")", " ", site.wait_time*100
        ind+=1
     
    total_cost = 1/ float(Fitness(final).routeFitness())
    w1 = 1
    w2 = 0
    total_time = float(Fitness(final).routeLength())
    w2 = 100
    total_time_wait = float(Fitness(final).routeLength()) 

    # write results 
    print "Total time taken : ", total_time, " with wait ", total_time_wait, "cost = ", total_cost
    with open("100runs_log.txt", "a") as myfile:
        myfile.write(str(len(final)) + " " + str(total_time) + " " + str(total_time_wait) + " " + str(total_cost) + "\n")

    # visualise in map
    xs = []
    ys = []
    for route in final:
        xs.append(route.x)
        ys.append(route.y)

    dpi = 80
    im = plt.imread('map_with_Goals.jpg')
    height, width, nbands = im.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im, interpolation='nearest')

    plt.scatter(xs, ys)
    i = 0
        
    for x, y in zip(xs, ys):
        i+=1
        plt.text(x, y, str(i), color="red", fontsize=30)
    fig.savefig('test_seq_demo.jpg', dpi=dpi, transparent=True)
    plt.show()
    q -= 1

print "wait_distribution ", wait_distribution
print "prob_distribution ", prob_distribution
