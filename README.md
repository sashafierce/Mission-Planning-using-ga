# Sensor based Robot Mission Planning using Genetic Algorithm


- Robot mission planning deals with accomplishing a mission that requires the robot to visit a set of sites and carrying out specific tasks at each site, consisting of Boolean and temporal constraints. 
- The Boolean constraints are like “Visit any one of three coffee machines” and “Visit any two of three instructors”, while temporal constraints typically ask the robot to carry some sub-missions in sequential orders and some others in parallel.
- An Evolutionary algorithm is devised to solve the problem, enabling the use of probabilistic optimality properties. 
- Each site is assumed to have an associated waiting time which is the average time for which the robot will have to wait in queue till the resource is available for use. A greedy heuristic is used to align the sites in the crossover operation to approach a better solution faster. 

`mission_planning.py` contains the implementation of proposed GA 
(please note the code is a bit shabby currently)

## Instuctions to run

```bash 
git clone https://github.com/sashafierce/Mission-Planning-using-ga
cd Mission-Planning-using-ga
pip install numpy matplotlib pandas 
python2.7 mission_planning.py
```

- `coord.txt` contains the site information (IIITA Robotics lab)
- `transition cost.txt`  contains the cost matrix (shortest viable path length for all lab site pairs)
- `100runs_log.txt` contains time taken to complete the mission after each run of the algorithm (total 100 runs)
- `plot_errorbars.py` is script that reads results file and plot error bars
- `mission_planning_optimistic.py` is the optimistic version of the approach for comparison
- `mission_planning_pessimistic.py` is the pessimistic version of the approach for comparison

## Experimental results
 
 Results confirm that the proposed solution performs well as compared to baseline optimistic and pessimistic approaches.

<img src ="https://user-images.githubusercontent.com/18103181/49593378-6edba100-f999-11e8-804f-92d94542152e.jpg">


### Dependencies 
1. Python 2.7
2. Numpy
3. Pandas
4. Matplotlib

## Sample mission solved
The robot is instructed by numerous people to complete a set of missions A to C. The numbers denote the mission sites. (A) Ross (9) wants his project report to be collected by the robot for getting signed by any 2 panelists out of three panelists namely, Bishnu (29), Shruti (28) and Rohan (21) , then to be submitted to either GCN (1) or Mary (4). (B) Meanwhile, the person Joey (6) wants to get his room keys which may be with any of this friend Rohan (21), Chandler(22) or Daksha (25). After getting the key the robot should return to him. (C) Rohan (21)  wants to get a stipend form printout from any of the machines copying_station (7) or office_pc (8) (either may be mal-functional) and after signing to be submitted to either of the supervisor office_rk (2) or GCN (1) for signature and finally given to office Mary (4).

### Solution by proposed approach
The output mission site sequence of the robot, Start from the robita entrance gate
1.  Go to cabin_ross (9 )  to collect project report (A)
2.  Next goto daksh (25)  to collect room keys for Joey(B), is not available
3.  Goto shruti (28) to get project report signed by panelists (A) 
4.  Visit rohan (21) to get project report signed by panelists (A) 
5.  Goto office_mary (4) to submit the signed doc (A)
6. Goto Rohan (21) to collect the stipend form (C) and to collect the room keys for Joey(B)
7. Goto cabin_joey (6)  again to finally give the collected keys (B)
8. Visit copying_station (7)  next to get the copy of the stipend form(C)
9. Goto office_rk (2)  to get the stipend form signed by supervisor(C)
10. Finally visit office_mary (4)  to submit the signed stipend form to office (C)
