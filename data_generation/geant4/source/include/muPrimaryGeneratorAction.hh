
#ifndef muPrimaryGeneratorAction_h
#define muPrimaryGeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4GeneralParticleSource.hh" // [yy] for gps
#include "globals.hh"

class G4ParticleGun;
//class G4ParticleTable; //[yy]

class G4Event;
class muDetectorConstruction;

class muPrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
    public:
        muPrimaryGeneratorAction(); // [yy] to declear constructor
        virtual ~muPrimaryGeneratorAction();
        virtual void GeneratePrimaries(G4Event*);
        //muPrimaryGeneratorAction(const muDetectorConstruction*);
    
    private:
        G4GeneralParticleSource* gpsParticleGun; // [yy] for gps
        //G4ParticleGun*            particleGun;
        //G4ParticleTable*			particleTable;
    
};


#endif


