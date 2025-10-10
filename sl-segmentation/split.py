import numpy as np

total_videos = 5995
test_size = int(0.1 * total_videos)
dev_size = int(0.1 * total_videos)

ids = np.array(range(total_videos))
np.random.seed(584)
np.random.shuffle(ids)

test_idx = np.sort(ids[:test_size])
dev_idx = np.sort(ids[test_size:test_size+dev_size])
train_idx = np.sort(ids[test_size+dev_size:])

np.savetxt('./split/test.csv', test_idx, delimiter=',')
np.savetxt('./split/dev.csv', dev_idx, delimiter=',')
np.savetxt('./split/train.csv', train_idx, delimiter=',')
