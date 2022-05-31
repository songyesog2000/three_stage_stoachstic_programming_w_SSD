# multicut algorithm for multi-stage stochastic programming with SSD constraints

### This is a working-on respostory
we eventual goal is to <br>
(1) establish a solver package implementing the multicut algorithm to generic multi-stage problem with SSD concerned. <br>

(2) standardize a data format/structure to store the multi-stage problem setting as well as the benchmark information for our SSD method. <br>

Currently, we have realized our algorithm on a toy problem ---- three-stage news-vendor problem:<br>

1) first stage:<br>
&emsp; vendors require a fix-investment from agents:<br>
        &emsp;&emsp;variable:<br>
            &emsp;&emsp; &emsp;p\in (0,10) - units of investment<br>
        &emsp;&emsp;coeffients:<br>
            &emsp;&emsp; &emsp; 2 - cost per unit of investment<br>
            
2) second stage<br>
    agents buy news paper from vendors before retailing it, <br>
    the available quantity depends on the fix investmentthe:<br>
        variable:<br>
            x \in (0,10*p)- units of news paper<br>
        coeffients:<br>
            -1 - cost per unit of news paper<br>
        random variable:<br>
3) third stage, agents retail news paper to individuals:<br>
&emsp; &emsp; variables:<br>
&emsp; &emsp; &emsp; z $\in (0,x)$, number of sold news paper<br>
&emsp; &emsp; random variables:<br>
&emsp; &emsp; &emsp; d $\in (0,\inf)$, demand<br> 

toy_two_stage_newsvendor_SSD showcased how SSD constrainted solution is different from the non-SSD constrainted solution

toy_three_stage_newsvendor_SSD is the toy we showcase our multi-cut algorithm

[1] Darinka Dentcheva, Mingsong Ye, Yunxuan Yi (2022). Risk-averse sequential decision problems with time-consistent stochastic
dominance constraints
