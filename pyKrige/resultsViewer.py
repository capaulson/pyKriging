__author__ = 'chrispaulson'
import pymongo


class pullResults():
    def __init__(self):
        self.databaseIP = 'localhost'
        # self.databaseIP = '152.78.142.2'
        self.client = pymongo.MongoClient(self.databaseIP)
        self.currentDB = self.client.currentDB
        self.target = self.currentDB.entry.find_one()['dbid']
        self.database = self.client[str(self.target)]

    def loadData(self):
        entries = self.database.entries
        x = []
        y = []
        for i in entries.find({'tracking.geometry.status':True}):
            x.append([i['geometryParameters']['ar'], i['geometryParameters']['area'], i['geometryParameters']['sweep']])
            y.append(i['Results']['geometry']['optimalChord'])
            # print i['geometryParameters']['ar'], i['geometryParameters']['area'], i['geometryParameters']['sweep'],i['Results']['geometry']['refArea']*2, i['Results']['geometry']['optimalAR'], i['Results']['geometry']['optimalScale'], i['Results']['geometry']['optimalChord']
        return x,y

    def printAll(self):
        entries = self.database.entries
        for i in entries.find({'tracking.geometry.status':True}):
            print i['geometryParameters']['ar'], i['geometryParameters']['area'], i['geometryParameters']['sweep'],i['Results']['geometry']['refArea']*2, i['Results']['geometry']['optimalAR'], i['Results']['geometry']['optimalScale'], i['Results']['geometry']['optimalChord']

if __name__=='__main__':
    a = pullResults()
    a.printAll()
