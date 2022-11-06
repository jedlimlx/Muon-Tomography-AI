
#ifndef muSensorHit_h
#define muSensorHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"

class muSensorHit : public G4VHit
{
public:
    
    muSensorHit();
    ~muSensorHit();
    muSensorHit(const muSensorHit&);
    const muSensorHit& operator=(const muSensorHit&);
    G4int operator==(const muSensorHit&) const;
    
    inline void* operator new(size_t);
    inline void  operator delete(void*);
    
    void Draw();
    void Print();
    
public:
    
    void Set(int event, int copy, const G4Track* track ,G4double eLoss, G4double valEIn);
    
    G4int GetEventNO()            const  { return eventNO; }; // [yy]
    G4int GetCopyNO()             const  { return copyNO; };
    G4int GetTrackID()            const  { return trackID; };
    G4int GetPDGcode()            const  { return codePDG; };
    G4double GetCharge()          const  { return charge; };      
    G4double GetEnergy()          const  { return energy; };      
    const G4ThreeVector& GetMomentum() const  { return momentum; };
    const G4ThreeVector& GetPos()      const  { return pos; };
    G4double GetTime()                 const  { return time; };      
    G4double GetEdep() const { return eDep;};
    void AddEdep(G4double val){ eDep += val;};
    G4double GetEIn() const { return eIn;};
    
private:
    G4int         eventNO; // [yy]
    G4int         copyNO;
    G4int         trackID;
    G4int         codePDG;
    G4double      charge;
    G4double      energy;
    G4ThreeVector momentum;
    G4ThreeVector pos;
    G4double      time;
    G4double 	  eDep;
    G4double      eIn;
};


typedef G4THitsCollection<muSensorHit> muSensorHitsCollection;

extern G4Allocator<muSensorHit> muSensorHitAllocator;


inline void* muSensorHit::operator new(size_t)
{
    void *aHit;
    aHit = (void *) muSensorHitAllocator.MallocSingle();
    return aHit;
}


inline void muSensorHit::operator delete(void *aHit)
{
    muSensorHitAllocator.FreeSingle((muSensorHit*) aHit);
}


#endif
