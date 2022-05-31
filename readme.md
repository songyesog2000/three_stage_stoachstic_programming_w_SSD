# multicut algorithm for multi-stage stochastic programming with SSD constraints

### This is a working-on respostory
Our eventual goal is to <br>
(1) establish a solver package implementing the multicut algorithm we proposed to generic multi-stage problem with SSD concerned [1]. <br>

(2) standardize a data format/structure to store the multi-stage problem setting as well as the benchmark information for our SSD method. <br>

Currently, we have realized our algorithm on a toy problem ---- three-stage news-vendor problem:<br>

1) First stage: agents require certain units of fix-investment from agents <br>
&emsp;variable:<br>
&emsp;&emsp; &emsp;$ q\in (0,10)$ - units of investment<br>
&emsp;coeffients:<br>
&emsp;&emsp; &emsp; 0.3 - cost per unit of investment<br>
            
2) Second stage: vendor buys news paper from agent, the available quantity range depends on the invesment <br>
&emsp;  variable:<br>
&emsp; &emsp; &emsp; $x \in (0,\inf)$ - units of news paper<br>
&emsp; coeffients:<br>
&emsp; &emsp; &emsp;-1 - cost per unit of news paper<br>
&emsp; random variable:<br>
&emsp; &emsp; &emsp; $\mathrm{cap}$ - the availble buying quantity is upper bounded by $\mathrm{cap}*q$

3) Third stage: vendor sells news paper to individuals<br>
&emsp;  variables:<br>
&emsp; &emsp; &emsp; $z \in (0,x)$, sold quantity of news paper<br>
&emsp;  random variables:<br>
&emsp; &emsp; &emsp; $d \in (0,\inf)$, quantity of demands<br> 

toy_two_stage_newsvendor_SSD showcased how SSD constrainted solution is different from the non-SSD constrainted solution

toy_three_stage_newsvendor_SSD is the toy we showcase our multi-cut algorithm

[1] Darinka Dentcheva, Mingsong Ye, Yunxuan Yi (2022). Risk-averse sequential decision problems with time-consistent stochastic
dominance constraints