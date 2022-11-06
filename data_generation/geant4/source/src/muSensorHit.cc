
#include "muSensorHit.hh"
#include "G4UnitsTable.hh"
#include "G4VVisManager.hh"
#include "G4Circle.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"

#include "G4SystemOfUnits.hh"

G4Allocator<muSensorHit> muSensorHitAllocator;


muSensorHit::muSensorHit():
eventNO(-1), // [yy]
copyNO(-1),
trackID(0),
codePDG(0),
charge(0.0),
energy(0.0),
time(0.0),
eDep(0.0),
eIn(0.0)
{}


muSensorHit::~muSensorHit() 
{}



muSensorHit::muSensorHit(const muSensorHit& right)
: G4VHit()
{
    eventNO     = right.eventNO; // [yy]
    copyNO      = right.copyNO;
    trackID     = right.trackID;
    codePDG     = right.codePDG;
    charge      = right.charge;
    energy      = right.energy;
    momentum    = right.momentum;
    pos         = right.pos;
    eDep 		= right.eDep;
    eIn       =right.eIn;
    
}


const muSensorHit& muSensorHit::operator=(const muSensorHit& right)
{
    if (this != &right) {
        eventNO     = right.eventNO; //[yy]
        copyNO      = right.copyNO;
        trackID     = right.trackID;
        codePDG     = right.codePDG;
        charge      = right.charge;
        energy      = right.energy;
        momentum    = right.momentum;
        pos         = right.pos;
        eDep 		= right.eDep;
        eIn       =right.eIn;
        
    }
    return *this;
}



G4int muSensorHit::operator==(const muSensorHit& right) const
{
    return (this==&right) ? 1 : 0;
}


void muSensorHit::Set(int event, int copy, const G4Track* track,G4double eLoss, G4double valEIn)
{
    eventNO  = event; //[yy]
    copyNO   =  copy;
    eDep	 = eLoss;
    eIn      =  valEIn;
    trackID  =  track->GetTrackID();
    codePDG  =  track->GetDefinition()->GetPDGEncoding();
    charge   =  track->GetDefinition()->GetPDGCharge();
    energy   =  track->GetKineticEnergy();
    momentum =  track->GetMomentum();
    pos      =  track->GetPosition();
    if(time > track->GetGlobalTime()){
        time     =  track->GetGlobalTime();
    }
}

void muSensorHit::Draw()
{
    G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();
    if(pVVisManager){
        G4Circle circle(pos);
        circle.SetScreenSize(2.);
        circle.SetFillStyle(G4Circle::filled);
        G4Colour colour(1.,0.,0.);
        G4VisAttributes attribs(colour);
        circle.SetVisAttributes(attribs);
        pVVisManager->Draw(circle);
    }
}



void muSensorHit::Print()
{
    G4cout << "Event Number: " << eventNO << G4endl; // [yy]
    G4cout << "Copy Number: " << copyNO << G4endl; 
    G4cout << "TrackID: " << trackID << G4endl; 
    G4cout << "PDG code: "<< codePDG << "  "
    << "charge: " << charge/eplus << G4endl;
    G4cout << "enregy: " << energy /MeV  << G4endl;
    G4cout << "momentum: "<< momentum.x()/MeV << " "
    << momentum.y()/MeV << " "
    << momentum.z()/MeV << G4endl;
    G4cout << "position: "<< pos.x()/mm << " "
    << pos.y()/mm << " "
    << pos.z()/mm << G4endl;
    G4cout << "time: "<< time/ns << G4endl;
    G4cout << "Energy deposit" << eDep/MeV << G4endl;
}



