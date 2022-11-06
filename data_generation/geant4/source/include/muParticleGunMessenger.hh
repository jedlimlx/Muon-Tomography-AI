
#ifndef muParticleGunMessenger_h
#define muParticleGunMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "muParticleGun.hh"

class muParticleGun;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;


class muParticleGunMessenger: public G4UImessenger
{
public:
    muParticleGunMessenger();
    explicit muParticleGunMessenger(muParticleGun*);
    ~muParticleGunMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);
    G4String GetCurrentValue(G4UIcommand* command);
private:
    muParticleGun*		muPG;
    G4UIdirectory*			cmdDir;
    G4UIcmdWithAnInteger*		vtxCmd;
    G4UIcmdWithAnInteger*		parCmd;
    G4UIcmdWithADoubleAndUnit*  eneCmd;
    
};

#endif

