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
![Image](https://github.com/Adalyne/Genetic-Alog./blob/242dab2529fcd3bee041f4ff1589b26a6c2b4a36/Image/%E5%96%AE%E9%BB%9E%E4%BA%A4%E9%85%8D.png#50)    
### 雙點交配  
![Image](https://github.com/Adalyne/Genetic-Alog./blob/c55821212283a2a1ea25c3b21e6a91865ed3f6be/Image/%E9%9B%99%E9%BB%9E%E4%BA%A4%E9%85%8D.png)  
### 均勻交配  
![Image](https://github.com/Adalyne/Genetic-Alog./blob/5acf3806f071bfd6c87ae7cf43ac9276ab26aacc/Image/%E5%9D%87%E5%8B%BB%E4%BA%A4%E9%85%8D.png)  

6. 邊界處理 : 若超出定範圍直接拉回邊界。
7. Mutation : 首先要定義一突變率（Mutation rate），於過程中均勻分配產生一機率值，機率值小於突變率則需進行突變。這裡的突變是選兩個bit進行交換。

## 範例
球體目標函數：(x-x0)^2 + (y-y0)^2 + (z-z0)^2 = r^2   
球心：(x0, y0, z0) = (6, 7, 8), 半徑 r = 100  
找範圍 x：-60~60, y：-50~50, z：-70~70 內最小值  
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
