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
        # words
        self.wordList = json_content["words"][:CUTOFF]
        self.word_len = len(self.wordList)
        
        # 构造后的句子
        self.sentence = " ".join(["[CLS]"] + self.wordList + ["[SEP]"])
        
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
        golden_list = {}
        for entityJson in entitiesJson:
            start = entityJson["start"]
            if start >= CUTOFF:
                continue
                
            end = min(entityJson["end"], CUTOFF)
            etype = entityJson["entity-type"].split(":")[0]
            entity = " ".join(self.wordList[start:end])
            golden_list[entity] = {
                "start": start,
                "end": end,
                "type": etype
            }
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
                start = argumentJson["start"]
                end = min(argumentJson["end"], CUTOFF)
                entity = " ".join(self.wordList[start:end])
                value = (start, end, entity, pretty_str(argumentJson["role"]))
                values.append(value)
                
            golden_dict[key] = list(sorted(values))
        return golden_dict

    def generateTriggerLabelList(self, triggerJsonList):
        # 针对原来每个词做trigger
        triggerLabel = ["O" for _ in range(self.word_len)]

        def assignTriggerLabel(index, label):
            if index >= CUTOFF:
                return
            triggerLabel[index] = pretty_str(label)

        for eventJson in triggerJsonList:
            triggerJson = eventJson["trigger"]
            start = triggerJson["start"]
            end = triggerJson["end"]
            if end - start != 1:
                continue
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
