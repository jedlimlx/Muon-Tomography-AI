
#include "muParticleGun.hh"
#include "muParticleGunMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include <iostream>


muParticleGunMessenger::muParticleGunMessenger()
:muPG(0)
{
}

muParticleGunMessenger::muParticleGunMessenger(muParticleGun* pg)
:G4UImessenger(),
muPG(pg)
{
    
    
    cmdDir = new G4UIdirectory("/gun/usr/");
    cmdDir->SetGuidance("UI commands for primary generator");
    
    vtxCmd = new G4UIcmdWithAnInteger("/gun/usr/vtx",this);
    vtxCmd->SetGuidance("Select vertex 0:User 1:Top 2:centre 3:Random flux");
    vtxCmd->SetParameterName("vtx",false);
    vtxCmd->SetRange("vtx >= 0 && vtx <= 3");
    vtxCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
    
    parCmd = new G4UIcmdWithAnInteger("/gun/usr/particle",this);
    parCmd->SetGuidance("Select particle 0:User 1:neutron 2:mu- 3:Random flux neutron 4:Real neutron spectrum 5:fission(Cf) 6:positron(Na-22), 7:2gamma(Ps), 8:3gamma(Ps) 9:gamma 1275keV");
    parCmd->SetParameterName("part",false);
    parCmd->SetRange("part >= 0 && part <= 9");
    parCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
    
    eneCmd = new G4UIcmdWithADoubleAndUnit("/gun/usr/energy",this);
    eneCmd->SetGuidance("set energy for neutron flux");
    eneCmd->SetParameterName("ene",false);
    eneCmd->SetRange("ene > 0.");
    eneCmd->AvailableForStates(G4State_PreInit,G4State_Idle);
    
}

muParticleGunMessenger::~muParticleGunMessenger()
{
    delete vtxCmd;
    delete parCmd;
    delete eneCmd;
    delete cmdDir;
    
}

void muParticleGunMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
    if (command == vtxCmd ){
        muPG->positionFlag = vtxCmd->GetNewIntValue(newValue);
    }else if ( command == parCmd){
        muPG->particleFlag = parCmd->GetNewIntValue(newValue);
    }else if ( command == eneCmd){
        muPG->monoEnergy = eneCmd->GetNewDoubleValue(newValue);
    }
    
}

G4String muParticleGunMessenger::GetCurrentValue(G4UIcommand* command)
{
    G4String cv;
    
    if (command == vtxCmd ){
        cv = vtxCmd->ConvertToString(muPG->positionFlag);
    }else if ( command == parCmd){
        cv = parCmd->ConvertToString(muPG->particleFlag);
    }else if ( command == eneCmd){
        cv = eneCmd->ConvertToString(muPG->monoEnergy, "cm");
    }
    
    return cv;
    
}
