#Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

#Data

Aabefore = np.array([199,-32,135,131,184,120,63,159,167,88,166,-84,137,49,108,133,7,81,91,-102,-12,53,65,141,123,117])
Aaafter = np.array([225,-7,155,135,218,138,35,187,167,93,187,-60,164,99,108,116,1,61,111,-92,-16,91,73,148,123,117])
Az = np.array([92,215,133,105,179,238,171,63,21,68,99,149,52,197,141,66,116,56,59,67,135,84,145,49,126,122])
Avg = np.add(Aabefore,Aaafter)/2

#Stats

Diff = np.subtract(Aabefore,Aaafter)
final = (np.abs(Diff))
avgAvg = np.mean(Avg)


print('The maximum difference in score chagne is',(np.max(final)), 'which is Whats going on')

sumbefore = np.sum(Aabefore)
sumafter = np.sum(Aaafter)
azsum = np.sum(Az)
avgsum = np.sum(Avg)

print('My sum before relistening is ',sumbefore,'and my sum after relistening is', sumafter)
print('Azhurel sum is', azsum)

meanbefore = np.mean(Aabefore)
meanafter = np.mean(Aaafter)
Azavg = np.mean(Az)
avgmean = np.mean(Avg)

print('My mean before and after is ',meanbefore,'and', meanafter)
print('Azhurels mean is ', np.mean(Az))

stdbefore = np.std(Aabefore)
stdafter = np.std(Aaafter)
avgstd = np.std(Avg)

print('My standard deviation before and after is',stdbefore,'and', stdafter)
print('Azhurel standard deviation is', np.std(Az))


rangebef = np.abs(np.max(Aabefore)-np.min(Aabefore))

medianbefore = np.median(Aabefore)
medianafter = np.median(Aaafter)
avgmedian = np.median(Avg)

print('My median before and after is',medianbefore,'and',medianafter)
print('Azhurel median is', np.median(Az))


q1b = np.percentile(Aabefore, 25)
q2b = np.percentile(Aabefore, 50)
q3b = np.percentile(Aabefore, 75)

print("1 quarter of my scores are below", q1b)
print("Half of my scores are below", q2b)
print("3 quarters of my scores are below", q3b)



q1a = np.percentile(Aaafter, 25)
q2a = np.percentile(Aaafter, 50)
q3a = np.percentile(Aaafter, 75)

print("1 quarter of my scores are below", q1a)
print("Half of my scores are below", q2a)
print("3 quarters of my scores are below", q3a)


q1az = np.percentile(Az, 25)
q2az = np.percentile(Az, 50)
q3az = np.percentile(Az, 75)

print("1 quarter of Az scores are below", q1az)
print("Half of Az scores are below", q2az)
print("3 quarters of Az scores are below", q3az)


#Figures

plt.scatter(range(1,len(Aabefore)+1),Aabefore, c='g', label='Before')
plt.scatter(range(1,len(Aaafter)+1),Aaafter, c='r', label = 'After')
plt.xlabel('Ep #')
plt.ylabel('Score')
plt.title('Aaron Scores')
plt.xticks(range(len(Aabefore)+1))
plt.xlim(0.5,26.5)
plt.legend()

for i in range(len(Aabefore)):
    plt.vlines(i+1, ymin=-120, ymax=260, linestyle='dashed', alpha=0.13)

plt.savefig('Aaron Scores.png')
plt.show()

plt.scatter(range(1,len(Aabefore)+1),Avg, c='k', label='Aaron Final Scores')
plt.xlabel('Ep #')
plt.ylabel('Score')
plt.title('Final Scores')
plt.xticks(range(len(Aabefore)+1))
plt.xlim(0.5,26.5)
plt.legend()

for i in range(len(Aabefore)):
    plt.vlines(i+1, ymin=-120, ymax=260, linestyle='dashed', alpha=0.13)

plt.savefig('Final.png')    
plt.show()


plt.scatter(range(1,len(Az)+1),Az, c='b')
plt.title('Azhurel Scores')
plt.xlabel('Ep #')
plt.ylabel('Score')
plt.xticks(range(len(Aabefore)+1))
plt.xlim(0.5,26.5)

for i in range(len(Aabefore)):
    plt.vlines(i+1, ymin=-120, ymax=260, linestyle='dashed', alpha=0.13)

plt.savefig('Azhurel Scores.png')    
plt.show()

plt.scatter(range(1,len(Aabefore)+1),Avg, c='g', label='Aaron')
plt.scatter(range(1,len(Aaafter)+1),Az, c='b', label = 'Azhurel')
plt.xlabel('Ep #')
plt.ylabel('Score')
plt.title('Comparison')
plt.xticks(range(len(Aabefore)+1))
plt.xlim(0.5,26.5)
plt.legend()

for i in range(len(Aabefore)):
    plt.vlines(i+1, ymin=-120, ymax=260, linestyle='dashed', alpha=0.13)

plt.savefig('Comparison.png')    
plt.show()




import seaborn as sns

# Calculate skewness
skew_before = skew(Aabefore)
skew_after = skew(Aaafter)

# Plot histograms with density curve and mean line
sns.histplot(Avg, kde=True, stat='density', color='blue', alpha=0.5, label='Aaron')

# Add vertical lines for the mean value
plt.axvline(x=np.mean(Avg), color='blue', linestyle='--')


# Set plot labels and legend
plt.xlabel('Scores')
plt.ylabel('Density')
plt.legend()

# Show plot
plt.savefig('Aaron Skewness.png')
plt.show()


# Plot histogram with density curve and mean line
sns.histplot(Az, kde=True, stat='density', color='red', alpha=0.5, label='Azhurel')

# Add vertical line for the mean value
plt.axvline(x=np.mean(Az), color='red', linestyle='--')

# Set plot labels and legend
plt.xlabel('Scores')
plt.ylabel('Density')
plt.xlim(-125,250)
plt.legend()

# Show plot
plt.savefig('Azhurel Skewness')
plt.show()

#Median Absolute Deviation


def mad(data, axis=None):
    # Median of data
    med = np.median(data, axis=axis)
    
    # Calculate deviations from the median
    deviations = np.abs(data - med)
    
    # Calculate the median of the deviations
    mad = np.median(deviations, axis=axis)
    
    return mad




mad_avg = mad(Avg)
mad_az = mad(Az)

print("MAD for Avg:", mad_avg)
print("MAD for Az:", mad_az)




# Show Data

print(Aabefore)
print(Aaafter)
print(Avg)
print(Az)


# z score

def zscore(x,mean,std):
    z = (x-mean)/std
    return z


zscore(43,113,55)


# Differences

ourdiff = np.subtract(Avg,Az)
totaldiff = np.abs(ourdiff)

plt.scatter(range(1,len(Az)+1),totaldiff, c='y')
plt.xlabel('Ep #')
plt.ylabel('Difference')
plt.title('Difference in Scores')
plt.xticks(range(len(Aabefore)+1))
plt.xlim(0.5,26.5)

for i in range(len(Aabefore)):
    plt.vlines(i+1, ymin=0, ymax=260, linestyle='dashed', alpha=0.13)

plt.savefig('difference.png')    
plt.show()

rangebef = np.abs(np.max(Aabefore)-np.min(Aabefore))
rangeaf =  np.abs(np.max(Aaafter)-np.min(Aaafter))
rangeav =  np.abs(np.max(Avg)-np.min(Avg))
rangeaz =  np.abs(np.max(Az)-np.min(Az))


print(rangebef)
print(rangeaf)
print(rangeav)
print(rangeaz)