
#include "muSensorSD.hh"
#include "G4HCofThisEvent.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "G4SDManager.hh"
#include "G4ios.hh"
#include "G4EventManager.hh" // [yy]
#include "G4Event.hh"  // [yy]

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "muAnalyzer.hh"

#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"

/////////////////////////////////////////////////////

muSensorSD::muSensorSD(G4String name)
:G4VSensitiveDetector(name)
{
    //G4String HCname;
    collectionName.insert("sensorCollection");
  
    eThreshold = 10.0 * eV;
    
}

/////////////////////////////////////////////////////

muSensorSD::~muSensorSD()
{ 
}

void muSensorSD::SetAnalyzer(muAnalyzer* analyzer_in){
    analyzer = analyzer_in;
}

/////////////////////////////////////////////////////

void muSensorSD::Initialize(G4HCofThisEvent* HCE)
{
    sensorCollection = new muSensorHitsCollection
    (SensitiveDetectorName,collectionName[0]);
    static G4int HCID = -1;
    if(HCID<0)
    { HCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]); }
    HCE->AddHitsCollection( HCID, sensorCollection );
    
}

/////////////////////////////////////////////////////

G4bool muSensorSD::ProcessHits(G4Step* aStep,G4TouchableHistory*)
{
    
    const G4Track * aTrack =  aStep->GetTrack();
    
    //if ( aTrack->GetDefinition()->GetPDGCharge() == 0.0) return false;
    //G4cout << aTrack->GetDefinition()->GetPDGEncoding() << G4endl;
    // Sensitive to neutron only
    //if ( aTrack->GetDefinition()->GetPDGEncoding() != 2112) return false;
    
    //Check Energy deposit
    G4double eLoss = aStep->GetTotalEnergyDeposit();
    //if (eLoss <= 0.0 ) return false;
    G4double time = aTrack->GetGlobalTime();
    G4int eventNO = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID(); //[yy]
    G4int copyNO = aStep->GetPreStepPoint()->GetTouchableHandle()->GetCopyNumber();
    G4int NbHits = sensorCollection->entries();
    G4bool found = false;
    for (G4int iHit=0; (iHit<NbHits) && (!found) ;iHit++) {
        found = (copyNO == (*sensorCollection)[iHit]->GetCopyNO() ) ;
        if (found) {
            (*sensorCollection)[iHit]->AddEdep(eLoss);
            return true;
        }
    }
    
    muSensorHit* newHit = new muSensorHit();
    G4double eIn = aStep->GetPreStepPoint()->GetKineticEnergy();
    newHit->Set(eventNO, copyNO, aTrack, eLoss, eIn);
    sensorCollection->insert( newHit );
    
    return true;
}

///////////////////////////////////////////////////////

void muSensorSD::EndOfEvent(G4HCofThisEvent* HCE)
{
    G4int NbHits = sensorCollection->entries();
    if (verboseLevel>0) {
        G4cout << "\n-------->Hits Collection: in this event they are "
        << NbHits
        << " hits in the tracker chambers: " << G4endl;
    }
    
    G4bool isFirstHit = true;
    G4bool hasHit=false;
    
    std::vector<G4int> event_out; // [yy]
    std::vector<G4double> x_out;
    std::vector<G4double> y_out;
    std::vector<G4double> z_out;
    std::vector<G4double> time_out;
    std::vector<G4double> eIn_out;
    std::vector<G4double> eDep_out;
    std::vector<G4int> trackID_out;
    std::vector<G4int> copyNo_out;
    std::vector<G4int> particleID_out;
    
    for (G4int i=0;i<NbHits;i++){
        muSensorHit* hit = (*sensorCollection)[i];
        if (verboseLevel>1) hit->Print();
        
        // output hits other than trigger counters
        //if (hit->GetEdep() < eThreshold ) continue;
        if (isFirstHit) {
            isFirstHit = false;
            hasHit = true;
        }

        event_out.push_back(hit->GetEventNO());
        x_out.push_back(hit->GetPos().x()/mm);
        y_out.push_back(hit->GetPos().y()/mm);
        z_out.push_back(hit->GetPos().z()/mm);
        time_out.push_back(hit->GetTime()/ns);
        eIn_out.push_back(hit->GetEIn()/MeV);
        eDep_out.push_back(hit->GetEdep()/MeV);
        trackID_out.push_back(hit->GetTrackID());
        copyNo_out.push_back(hit->GetCopyNO());
        particleID_out.push_back(hit->GetPDGcode());
    }
    
    if (NbHits!=0){ // [yy]
    //analyzer->Fill(NbHits,x_out,y_out,z_out,time_out,eIn_out,eDep_out,trackID_out,copyNo_out,particleID_out);
    analyzer->Fill(NbHits,event_out,x_out,y_out,z_out,time_out,eIn_out,eDep_out,trackID_out,copyNo_out,particleID_out); // [yy]
    }
    
}
