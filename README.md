# Genetic-Alog.(GA)
基因演算法是一種優化演算法，主要依據達爾文“進化論”中所提出的「物競天擇，適者生存，不適者淘汰」的生物演化法則而來，其核心觀念為模擬基因的優勝劣汰，以進行最佳化的計算。  
## GA 背景((Background)
基因演算法 (GA) 的架構一開始是由 John Holland 教授於1975年所提出，該演算法的主要靈感來自於生物圈中的演化機制，在大自然中，生物的繁衍及遺傳是透過染色體的交配與變異，以改變不同基因的組成，來產生下一代，染色體主要是由 DNA 及蛋白質所組成，一段 DNA 片段代表著控制某一性狀的基因，因此染色體也表示是由許多基因所組成。

簡單來說，基因演算法即是透過這種概念所發展，將求解問題的一個潛在解或參數用一個染色體來表示，藉由編碼將染色體轉成字串或數值等形式，而每個數值或字串代表染色體內的基因，表示解的某個部分，接著透過突變及交配的方式，來產生下一代，也就是不同的潛在解，最後以適者生存，不適者淘汰的觀念，將好的解進行保留，以進行下一輪的交配突變，來產生更好的解，直到所設定的停止條件到達為止，期望未來能夠跳脫局部解找到全域最佳解。

## GA演算法
GA主要求最佳化解。  
**參數定義及介紹**   
`N`：Initial population 的大小，此變數決定初始群集內chromosome的個數。  
`D`：Dimension 維度大小，欲求解目標問題之變數數量。 (ex. ax + b = y 求a b的值符合最佳解，所以D = 2)  
`B`：Bits number，D 維度中每個變數可能出現的數字受限於 bit 數的控制。(ex. 0<a<50, 0<b<100, 因為2^7=128,所以B=7)  
`n`：於每次 Iteration 中從 population 抓取菁英個數的數量。  
`cr`：Crossover rate，交配之門檻值。  
`mr`：Mutation rate，突變之門檻值。  
`max_iter`：欲進行迭代之總次數。  
**步驟**  
1. Initial population : 在給定範圍內初始化染色體及族群大小。
2. Fitness : 對染色體進行計算，fitness越小(誤差越小)表示染色體的基因優良，將來在Selection上選取優良的基因會越高。
3. Selection : 選出那些染色體進行交配，通常採用`輪盤法（Roulette wheel）`來選取菁英個體。輪盤法是一種回放式隨機取樣法，將一輪盤分成 N 個部分，根據 fitness 決定其盤面面積大小，fitness 越佳面積就越大，故在隨機取樣中被選到的機會就會越大。此部分會從初始群集中篩選出 n 組最佳 chromosome 當做父代。
![Image](https://github.com/Adalyne/Genetic-Alog./blob/312d4f908b6bc644be7211cfb595a6cf3a4a2ebd/Image/%E8%BC%AA%E7%9B%A4%E6%B3%95.png)  
4. Crossover : 先定義交配率，隨後以均勻分配產生一機率值，機率值小於交配率則需進行交配。採用交配方法為單點交配（One-point crossover）、雙點交配、均勻交配。
### 單點交配  
![Image](https://github.com/Adalyne/Genetic-Alog./blob/242dab2529fcd3bee041f4ff1589b26a6c2b4a36/Image/%E5%96%AE%E9%BB%9E%E4%BA%A4%E9%85%8D.png)  
### 雙點交配  
![Image](https://github.com/Adalyne/Genetic-Alog./blob/c55821212283a2a1ea25c3b21e6a91865ed3f6be/Image/%E9%9B%99%E9%BB%9E%E4%BA%A4%E9%85%8D.png)  
### 均勻交配  
![Image](https://github.com/Adalyne/Genetic-Alog./blob/5acf3806f071bfd6c87ae7cf43ac9276ab26aacc/Image/%E5%9D%87%E5%8B%BB%E4%BA%A4%E9%85%8D.png)  

6. 邊界處理 : 若超出定範圍直接拉回邊界。
7. Mutation : 首先要定義一突變率（Mutation rate），於過程中均勻分配產生一機率值，機率值小於突變率則需進行突變。這裡的突變是選兩個bit進行交換。

## 範例
球體目標函數：(x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2   
球心：(x0, y0, z0) = (6, 7, 8), 半徑 r = 100  
找範圍 x：-60-60, y：-50-50, z：-70-70 內最小值  
**步驟1**  
初始化需要的變數    
```ruby
def __init__(self, Number=10, Dimension=3, Bitnum=8, Elite_num=2, CrossoverRate=0.9, MutationRate=0.1, MaxIteration= 10000):
        self.N = Number    #Initial population 的大小
        self.D = Dimension    #Dimension 維度大小，欲求解目標問題之變數數量
        self.B = Bitnum      #D 維度中每個變數可能出現的數字受限於 bit 數的控制
        self.n = Elite_num    #於每次 Iteration 中從 population 抓取菁英個數的數量
        self.cr = CrossoverRate    #Crossover rate，交配之門檻值
        self.mr = MutationRate   #Mutation rate，突變之門檻值
        self.max_iter = MaxIteration    #欲進行迭代之總次數
```
**步驟2**  
初始化10組染色體(x,y,z)，在限制範圍內隨機生成10組(x,y,z)
```ruby
def generationPopulation(self):
        population = list()
        for number in range(self.N):
            chrom_list = list()
            X = np.random.randint(-60,61)
            Y = np.random.randint(-50,51)
            Z = np.random.randint(-70,71)
            chrom_list.append(X)
            chrom_list.append(Y)
            chrom_list.append(Z)
            population.append(chrom_list)
        return population
```
因為我的初始化是十進位，所以要寫一個function轉成2進位(可以順便寫一個將二進位轉成十進位，之後要計算fitness一定會用到)  
因為我的限制中有負數，所以要多一個bit紀錄正負數   
```ruby
def D2B(self, num):   #十進位轉二進位
        bit_map = list()
        if(num<0):
            bit_map.append(1)
            zero_num = self.B-len(bin(-num)[2:])-1
            for i in range(zero_num):
                bit_map.append(0)
            for i in bin(-num)[2:]:
                bit_map.append(int(i))
        else:
            bit_map.append(0)
            zero_num = self.B-len(bin(num)[2:])-1
            for i in range(zero_num):
                bit_map.append(0)
            for i in bin(num)[2:]:
                bit_map.append(int(i))
        return bit_map

def B2D(self, bitnum):   #二進位轉十進位
    dec = ''
    for i in range(1,self.B):
        dec += str(bitnum[i])
    if bitnum[0] == 0:
        return int(dec, 2)

    else:
        return -int(dec, 2)
```
**步驟3**  
對染色體計算fitness值  
因為我的目標函數為 (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2   ， 球心：(x0, y0, z0) = (6, 7, 8), 半徑 r = 100   
所以我的fitness函數為：r^2 - ((x-6)^2 + (y-7)^2 + (z-8)^2) 的平方  
```ruby
def fitness(self, pop):
        fit_list = list()
        (x0, y0, z0) = (6, 17, 8)
        r =100
        for n in range(len(pop)):
            fit = r**2 - (pop[n][0]-x0)**2 - (pop[n][1]-y0)**2 - (pop[n][2]-z0)**2
            fit_list.append(fit**2)
        return fit_list
```
**步驟4**  
Selection : 每次用`輪盤法`選出兩條染色體進行交配  
如何製作輪盤?  
1. 將finess_map的數全部相加
2. 1-(每個fitness_map/sum(fitness_map)) ， 這個公式表示越小的fitness在輪盤上的面積會越大
3. 然後正規化，將值對應到(0,1)中後，用累加的方式加入wheel list輪盤中
4. 以上就是輪盤的作法，然後在random(0-1)的數決定是在哪個區間
```ruby
def Selection(self, pop_bin, fitness):
        select_bin = pop_bin.copy()
        fitness1 = fitness.copy()
        Parents = list()
        if sum(fitness) == 0:
            for i in range(self.n):
                parent = select_bin[random.randint(0,self.N)-1]
                Parents.append(parent)
        else:
            #print('sum of fitness=',sum(fitness))
            wheel = [(1 - (fit_num/sum(fitness1)))/(self.N-1) for fit_num in fitness1]
            #print('wheel=',wheel)
            tep = 0
            Cumulist = list()
            Cumulist.append(tep)
            for i in range(len(wheel)):
                tep += wheel[i]
                Cumulist.append(tep)
            #print('Cumulist=',Cumulist)
            for i in range(self.n):
                z1 = random.uniform(0,1)
                #print('z1=',z1)
                for pick in range(len(Cumulist)-1):
                    if Cumulist[pick] <= z1 < Cumulist[pick+1]:
                        parent = select_bin[wheel.index(wheel[pick])]
                Parents.append(parent)
        return Parents
```
**步驟5**  
Crossover：我選擇的是`雙點交配`, 每次選擇兩組染色體進行交配   
交配率設0.9，若小於0.9代表交配成功  
交配完的children若超出限制，則拉回邊界線上  
Mutation : 10%的突變率讓染色體內兩個bit交換，若超出邊界則拉回邊界上    
```ruby
def Mutation(self, Children):
        mr_children = list()
        for child in Children:
            z1 = random.uniform(0,1)
            #print('mr=',z1)
            if(z1<0.1):
                element_index1 = random.randint(0,self.B-1)
                element_index2 = random.randint(0,self.B-1)
                #print('exchange index = (',element_index1,element_index2,')')
                temp = child[element_index1]
                child[element_index1] = child[element_index2]
                child[element_index2] = temp
            mr_children.append(child)
        return mr_children

def Crossover(self, Parents):
        #crossover
        def swap_gene(p1, p2, index1, index2):
            temp = p1[index1:index2]
            p1[index1:index2] = p2[index1:index2]
            p2[index1:index2] = temp
            #print('p1=',p1,'p2=',p2)
            return p1, p2
        parents = Parents.copy()
        z1 = random.uniform(0,1)   #雖機生成交配率
        #print('cr z1=',z1)
        if(z1<self.cr):
            z2 = random.randint(0,self.B-3)
            z3 = random.randint(z2,self.B-2)
            #print('index=', z2)
            child1 = list()
            child2 = list()
            for i in range(self.D):
                ch1, ch2 = swap_gene(Parents[0][i], Parents[1][i], z2, z3)
                child1.append(ch1)
                child2.append(ch2)
            parents = list()
            #print('Crossover=',child1,child2)
            child1 = self.Restrict(child1)
            child2 = self.Restrict(child2)
            #print('Restrict=',child1,child2)
            #Mutation
            child1 = self.Mutation(child1)
            child2 = self.Mutation(child2)
            #print('Mutation=',child1,child2)
            child1 = self.Restrict(child1)
            child2 = self.Restrict(child2)

            parents.append(child1)
            parents.append(child2)

        return parents
```
**步驟6**   
撰寫主程式
```ruby
def main():
    ga = GeneticAlogorithn()
    print('Population Size=',ga.N,' Population Dimension=',ga.D,' Bit Diamention=', ga.B)
    #initial
    pop_dec = ga.generationPopulation()
    print('dec population=',pop_dec)

    #encoding
    pop_bin = list()
    for i in range(ga.N):
        chrom_cv=list()
        for j in range(ga.D):
            chrom_cv.append(ga.D2B(pop_dec[i][j]))
        pop_bin.append(chrom_cv)
    print('bin populaation=',pop_bin)

    #fitness
    fitness_map = ga.fitness(pop_dec)
    print('fitness map=', fitness_map)

    best_rvlist = list()
    for e in range(ga.max_iter):
        best_fitness = min(fitness_map)
        print('i=',e,'best fitness=', best_fitness)
        best_rvlist.append(best_fitness)
        Parents = ga.Selection(pop_bin, fitness_map)
        Children = ga.Crossover(Parents)
        pop_bin.append(Children[0])
        pop_bin.append(Children[1])
        for i in range(len(Children)):
            child = list()
            for j in range(ga.D):
                child.append(ga.B2D(Children[i][j]))
            pop_dec.append(child)

        fitness_map = ga.fitness(pop_dec)
        for i in range(len(Children)):
            worst_fitness = max(fitness_map)
            pop_dec.pop(fitness_map.index(worst_fitness))
            pop_bin.pop(fitness_map.index(worst_fitness))
            fitness_map.pop(fitness_map.index(worst_fitness))

if __name__ == '__main__':
    main()
```
