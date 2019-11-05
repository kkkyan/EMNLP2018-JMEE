from enet.consts import CUTOFF


def pretty_str(a):
    a = a.upper()
    # Due to Yang 2015, All time-* change to time
    if a[:4] == "TIME":
        a = "TIME"
        
    if a == 'O':
        return a
    elif a[1] == '-':
        return a[:2] + "|".join(a[2:].split("-")).replace(":", "||")
    else:
        return "|".join(a.split("-")).replace(":", "||")


class Sentence:
    def __init__(self, json_content, graph_field_name="stanford-colcc"):
        # 原句子
        self.sentence = "[CLS] " + " ".join(json_content["words"][:CUTOFF]) + " [SEP]"
        # words
        self.wordList = json_content["words"][:CUTOFF]
        self.word_len = len(self.wordList)
        
        # trigger 标签， 针对words，去除了连续词的可能性
        self.triggerLabelList = self.generateTriggerLabelList(json_content["golden-event-mentions"])
        # entities
        self.entities = self.generateGoldenEntities(json_content["golden-entity-mentions"])
        # events
        self.events = self.generateGoldenEvents(json_content["golden-event-mentions"])
        self.containsEvents = len(self.events.keys())
        
        # token
        self.tokenList = self.makeTokenList()
        

    def generateGoldenEntities(self, entitiesJson):
        '''
        [(2, 3, "entity_type")]
        '''
        golden_list = []
        for entityJson in entitiesJson:
            start = entityJson["start"]
            if start >= CUTOFF:
                continue
                
            end = min(entityJson["end"], CUTOFF)
            etype = entityJson["entity-type"].split(":")[0]
            golden_list.append((start, end, etype))
        return golden_list

    def generateGoldenEvents(self, eventsJson):
        '''

        {
            (2, "event_type_str") --> [(1, 2, "role_type_str"), ...]
            ...
        }

        '''
        golden_dict = {}
        for eventJson in eventsJson:
            triggerJson = eventJson["trigger"]
            # 去除连续词的可能性
            if triggerJson["start"] >= CUTOFF or triggerJson["end"] - triggerJson["start"] != 1:
                continue
                
            key = (triggerJson["start"], pretty_str(eventJson["event_type"]))
            values = []
            for argumentJson in eventJson["arguments"]:
                if argumentJson["start"] >= CUTOFF:
                    continue
                value = (argumentJson["start"], min(argumentJson["end"], CUTOFF), pretty_str(argumentJson["role"]))
                values.append(value)
                
            golden_dict[key] = list(sorted(values))
        return golden_dict

    def generateTriggerLabelList(self, triggerJsonList):
        triggerLabel = ["O" for _ in range(self.word_len)]

        def assignTriggerLabel(index, label):
            if index >= CUTOFF:
                return
            triggerLabel[index] = pretty_str(label)

        for eventJson in triggerJsonList:
            triggerJson = eventJson["trigger"]
            start = triggerJson["start"]
            end = triggerJson["end"]
            etype = eventJson["event_type"]
            # 去除连续词
            if end - start != 1:
                continue
            assignTriggerLabel(start, etype)
            
        return triggerLabel

    def makeTokenList(self):
        return [Token(self.wordList[i], self.triggerLabelList[i]) for i in range(self.word_len)]

    def __len__(self):
        return self.word_len

    def __iter__(self):
        for x in self.tokenList:
            yield x

    def __getitem__(self, index):
        return self.tokenList[index]


class Token:
    def __init__(self, word, triggerLabel):
        self.word = word
        self.triggerLabel = triggerLabel
        self.predictedLabel = None

    def addPredictedLabel(self, label):
        self.predictedLabel = label
