
#include "muPrimaryGeneratorAction.hh"
#include "muParticleGun.hh"
#include "muDetectorConstruction.hh"

#include "G4Event.hh"
#include "G4MuonMinus.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4GeneralParticleSource.hh" // [yy] for gps

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

muPrimaryGeneratorAction::muPrimaryGeneratorAction()
 : G4VUserPrimaryGeneratorAction(), gpsParticleGun(0)
{

  gpsParticleGun = new G4GeneralParticleSource();

  // *** Note ***
  // If you fix the parameters here, you cannot change them from .mac file. I guess.

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

muPrimaryGeneratorAction::~muPrimaryGeneratorAction()
{

  delete gpsParticleGun;
  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void muPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{

  gpsParticleGun->GeneratePrimaryVertex(anEvent);
  
}
