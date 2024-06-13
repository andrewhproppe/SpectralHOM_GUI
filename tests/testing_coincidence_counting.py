from TimeTagger import createTimeTagger, Coincidence, Counter

tagger = createTimeTagger()
input_channels = [1, 2]

coincidences_vchannels = Coincidence(tagger, [1, 2], coincidenceWindow=10000)

channels = [*input_channels, *coincidences_vchannels.getChannels()]

counting = Counter(tagger, channels, 1e10, 300)
measurementDuration = 10e12 # 10 s
counting.startFor(measurementDuration)
counting.waitUntilFinished()

index = counting.getIndex()
counts = counting.getData()