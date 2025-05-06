# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readJSON(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        f.readline()
        for l in f:
            d = eval(l)
            u = d['userID']
            g = d['gameID']
            yield u, g, d

# %%
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# %%
gamesPerPlayer = defaultdict(list)
playersPerGame = defaultdict(list)
games = []
averageHours = 0
for player, game, data in allHours:
    gamesPerPlayer[player].append(game)
    playersPerGame[game].append(player)
    games.append(game)
    averageHours += data['hours_transformed']
averageHours = averageHours/len(allHours)

# %%
gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in hoursTrain:
  gameCount[game] += 1
  totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

games = list(gameCount.keys())

# %%
negativeValid = []
for player, game, data in hoursValid:
    randomGame = random.choice(games)
    while (True):
        if randomGame not in gamesPerPlayer[player]:
            break
        randomGame = random.choice(games)
    negativeValid.append((player, randomGame, {}))
hoursValid += negativeValid

# %%
def populationMinFinder (arg):
    top = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        top.add(i)
        if count > totalPlayed/arg: break
    correct = 0
    total = len(hoursValid)
    
    for player, game,_ in hoursValid:
        if game in gamesPerPlayer[player] and game in top:
            correct += 1
        elif game not in gamesPerPlayer[player] and game not in top:
            correct += 1
    accuracy = correct/total
    return accuracy, top

# %%
minArg = 2
minArgAccuracy = 0
for i in range(200):
    temp = 1.0 + 0.01*i
    if populationMinFinder(temp)[0] > minArgAccuracy:
        minArgAccuracy = populationMinFinder(temp)[0]
        minArg = temp
minArg

# %%
Accuracy, top = populationMinFinder(minArg)
Accuracy

# %%
#Jaccard
def CosineSet(s1, s2):
    # Not a proper implementation, operates on sets so correct for interactions only
    s1 = set(s1)
    s2 = set(s2)
    numer = len(s1.intersection(s2))
    denom = math.sqrt(len(s1)) * math.sqrt(len(s2))
    if denom == 0:
        return 0
    return numer / denom

def Jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom
    
def jaccardCompare(u, g):
    Jaccards = []
    for game in gamesPerPlayer[u]:
        if g == game: continue
        Jaccards += [Jaccard(playersPerGame[g], playersPerGame[game])]
    return numpy.mean(Jaccards)

# %%
gamesPerUserSort = defaultdict(list)
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u, g = l.strip().split(',')
    gamesPerUserSort[u].append((g, g in top, jaccardCompare(u, g))) 
    
for user, games_list in gamesPerUserSort.items():
    sorted_games = sorted(games_list, key=lambda x: (not x[1], -x[2]))
    gamesPerUserSort[user] = sorted_games

for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    sorted_games = gamesPerUserSort[u]
    index_of_g = next((i for i, game_tuple in enumerate(sorted_games) if game_tuple[0] == g), -1)
    pred = 1 if 0 <= index_of_g < len(sorted_games)//2 else 0
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# %%
gamesPerUserSort = defaultdict(list)
for u, g, d in hoursValid:
    gamesPerUserSort[u].append((g, g in top, jaccardCompare(u, g))) 

for user, games_list in gamesPerUserSort.items():
    sorted_games = sorted(games_list, key=lambda x: (not x[1], -x[2]))
    gamesPerUserSort[user] = sorted_games

correct3 = 0
total = len(hoursValid)

for u, g, d in hoursValid:
    sorted_games = gamesPerUserSort[u]
    index_of_g = next((i for i, game_tuple in enumerate(sorted_games) if game_tuple[0] == g), -1)
    pred = 1 if index_of_g < len(sorted_games)/2 else 0

    if g in gamesPerPlayer[u] and pred == 1:
        correct3 += 1
    elif g not in gamesPerPlayer[u] and pred == 0:
        correct3 += 1

accuracy = correct3 / total
print("Accuracy:", accuracy)


# %%
###Time Played Prediction

# %%
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)

gamesPerPlayer = defaultdict(list)
playersPerGame = defaultdict(list)
games = []
for player, game, data in hoursTrain:
    gamesPerPlayer[player].append(game)
    playersPerGame[game].append(player)
    games.append(game)

hoursPerUser = defaultdict(dict)
hoursPerItem = defaultdict(dict)

for player, game, data in hoursTrain:
    hoursPerUser[player][game] = data['hours_transformed']
    hoursPerItem[game][player] = data['hours_transformed']

betaU = {}
betaI = {}

for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0

alpha = globalAverage

def iterate(lamb):
    global alpha
    alpha = globalAverage
    for _ in range(100):
        # Update alpha
        numer = 0
        for u, g, d in hoursTrain:
            numer += hoursPerUser[u][g] - (betaU[u] + betaI[g])
        alpha = numer / len(hoursTrain) 
        # Update betaU for each user
        for u in betaU:
            numer = 0
            for g in gamesPerPlayer[u]:
                numer += hoursPerUser[u][g] - (alpha + betaI[g])
            # numer = numer 
            betaU[u] = numer / (lamb + len(gamesPerPlayer[u]))
        
        # Update betaI for each item
        for g in betaI:
            numer = 0
            for u in playersPerGame[g]:
                numer += hoursPerUser[u][g] - (alpha + betaU[u])
            # numer = numer 
            betaI[g] = numer / (lamb + len(playersPerGame[g]))

iterate(5) #ideal seems to be 5

# %%
def MSE(y, ypred):
    diffs = [(a-b)**2 for (a,b) in zip(y,ypred)]
    return sum(diffs) / len(diffs)

# %%
bestlambda = 0
bestMSE = 5
for i in range(20):
    y = []
    ypred = []
    print(i)
    iterate(4.8 + 0.01*i)

    for u, g, d in hoursValid:
        y += [d['hours_transformed']]
        ypred += [alpha + betaI[g] + betaU[u]]
    validMSE = MSE(y, ypred)

    if validMSE < bestMSE:
        bestMSE = validMSE
        bestlambda = 4.8 + 0.01*i
print('best lambda: ', bestlambda)
print('bestMSE: ', bestMSE)

# %%
iterate(4.95)

# %%
#Hw2 Rating Prediction Function
itemAverages = {}
ratingDict = {}
average = 0
count = 0

for d in allHours:
    user,item = d[0], d[1]
    ratingDict[(user,item)] = d[2]['hours_transformed']
    average += d[2]['hours_transformed']
    count += 1

average = average / count
for i in playersPerGame:
    rs = [ratingDict[(u,i)] for u in playersPerGame[i]]
    itemAverages[i] = sum(rs) / len(rs)

def predictRating(user,item):
    ratings = []
    similarities = []
    for i in gamesPerPlayer[user]:
        if i == item: continue
        ratings.append(ratingDict[(user, i)] - itemAverages[i])
        similarities.append(CosineSet(playersPerGame[item],playersPerGame[i])) #Jaccard
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # # User hasn't rated any similar items
        # if item in itemAverages:
        #     return itemAverages[item]
        # return alpha + betaU[user] + betaI[item]
        return average

y2 = []
ypred2 = []

for u, i, d in hoursValid:
    y2 += [d['hours_transformed']]
    ypred2 += [predictRating(u, i)]
validMSE2 = MSE(y2, ypred2)

validMSE2

# %%
#TensorFlow
userIDs = {}
itemIDs = {}
interactions = []

for u, i, d in allHours:
    r = d['hours_transformed']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i,r))

optimizer = tf.keras.optimizers.Adam(0.1) #was 0.1

class LatentFactorModel(tf.keras.Model):
    def __init__(self, mu, K, lamb):
        super(LatentFactorModel, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(gamesPerPlayer)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(playersPerGame)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(gamesPerPlayer),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(playersPerGame),K],stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i] +\
            tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p
    
    # Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2) +\
                            tf.reduce_sum(self.gammaU**2) +\
                            tf.reduce_sum(self.gammaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        pred = self.alpha + beta_u + beta_i +\
               tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)



# %%
modelLFM = LatentFactorModel(averageHours, 1, 0.0029) 


# %%
interactionsTrain = interactions[:165000]
def trainingStep(model, interactions):
    Nsamples = 164999
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

# %%
objPrev = 5
for i in range(100): #was 200
    obj = trainingStep(modelLFM, interactionsTrain)
    if obj < 2.42: #objPrev:
        break
    objPrev = obj
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

# %%
u,i,r = hoursValid[1]
modelLFM.predict(userIDs[u], itemIDs[i]).numpy()

y = []
ypred = []

for u, i, d in hoursValid:
    bu = betaU[u]
    bi = betaI[i]
    dot = tf.tensordot(modelLFM.gammaU[userIDs[u]], modelLFM.gammaI[itemIDs[i]], 1).numpy()

    y.append(d['hours_transformed'])
    ypred.append(alpha + bu + bi + dot)
validMSE = MSE(y, ypred)
validMSE

# %%
predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    bu = betaU[u]
    bi = betaI[g]
    tfPred = alpha + bu + bi + tf.tensordot(modelLFM.gammaU[userIDs[u]], modelLFM.gammaI[itemIDs[g]], 1).numpy()

    sorted_games = gamesPerUserSort[u]
    index_of_g = next((i for i, game_tuple in enumerate(sorted_games) if game_tuple[0] == g), -1)
    pred = 1 if 0 <= index_of_g < len(sorted_games)//2 else 0
    _ = predictions.write(u + ',' + g + ',' + str(tfPred) + '\n')

predictions.close()

# %%
averageHoursPerPlayer = defaultdict(float)
averageHoursPerGame = defaultdict(float)

for u, g, d in interactions:
    averageHoursPerPlayer[u] += d
    averageHoursPerGame[g] += d

for u in averageHoursPerPlayer:
    averageHoursPerPlayer[u] = averageHoursPerPlayer[u] / len(gamesPerPlayer[u])

for g in averageHoursPerGame:
    averageHoursPerGame[g] = averageHoursPerGame[g] / len(playersPerGame[g])


averageHoursPerGame


