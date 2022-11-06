
#ifndef muSensorSD_h
#define muSensorSD_h 1

#include "G4VSensitiveDetector.hh"
#include "muSensorHit.hh"
#include "muAnalyzer.hh"

#include <vector>

#include "TFile.h"
#include "TTree.h"

class G4Step;
class G4HCofThisEvent;


class muSensorSD : public G4VSensitiveDetector
{
public:
    muSensorSD(G4String);
    ~muSensorSD();
    
    muAnalyzer* analyzer;
    
    void Initialize(G4HCofThisEvent*);
    G4bool ProcessHits(G4Step*, G4TouchableHistory*);
    void EndOfEvent(G4HCofThisEvent*);
    
    G4double GetThresholdEnergy() const {return eThreshold;}
    G4double GetTimeResolution()  const {return tResolution;}
    
    static const G4String& GetCollectionName() {return HCname;}
    void SetAnalyzer(muAnalyzer*);
    
    
private:
    muSensorHitsCollection* sensorCollection;    
    G4double eThreshold;
    G4double tResolution;
    G4int event; // [yy]
    
    static const G4String HCname;
    
};


#endif

