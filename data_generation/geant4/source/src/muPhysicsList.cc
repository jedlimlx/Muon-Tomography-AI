
#include "muPhysicsList.hh"

#include "G4ProcessManager.hh"
#include "G4ParticleTypes.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

// [yy]
#include "globals.hh"
#include "G4EmStandardPhysics.hh" // 電磁相互作用
#include "G4LossTableManager.hh" 
#include "G4Decay.hh"
#include "G4DecayTable.hh"
#include "G4DecayWithSpin.hh"
#include "G4MuonDecayChannelWithSpin.hh"
#include "G4MuonRadiativeDecayChannelWithSpin.hh"

// gamma interactions
#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
// electron interactions
#include "G4eMultipleScattering.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"
// muon interactions
#include "G4MuMultipleScattering.hh"
#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"
// pion interactions
#include "G4hMultipleScattering.hh"
#include "G4hIonisation.hh"
#include "G4hBremsstrahlung.hh"
#include "G4hPairProduction.hh"

#include "G4ComptonScattering.hh"
#include "G4GammaConversion.hh"
#include "G4PhotoElectricEffect.hh"
#include "G4eMultipleScattering.hh"
#include "G4MuMultipleScattering.hh"
#include "G4hMultipleScattering.hh"
#include "G4eIonisation.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eplusAnnihilation.hh"
#include "G4MuIonisation.hh"
#include "G4MuBremsstrahlung.hh"
#include "G4MuPairProduction.hh"
#include "G4hIonisation.hh"
//#include "G4UserSpecialCuts.hh"
#include "G4StepLimiter.hh"

muPhysicsList::muPhysicsList():  G4VUserPhysicsList()
{
    defaultCutValue = 0.1*mm;
    SetVerboseLevel(1);
}


muPhysicsList::~muPhysicsList()
{}


void muPhysicsList::ConstructParticle()
{
    // In this method, static member functions should be called
    // for all particles which you want to use.
    // This ensures that objects of these particle types will be
    // created in the program. 
    
    ConstructBosons();
    ConstructLeptons();
    ConstructMesons();
    ConstructBaryons();

  // ********************************************************************************
  // Task 3.b - Exercise 7 
  //   to activate polarisation
  //   -  uncomment the lines below
  //   -  change the decay process in  PhysicsList::ConstructDecay() to
  //      G4DecayWithSpin
  // ********************************************************************************
  //
  
  G4DecayTable* MuonPlusDecayTable = new G4DecayTable();
  MuonPlusDecayTable -> Insert(new G4MuonDecayChannelWithSpin("mu+",0.986));
  MuonPlusDecayTable -> Insert(new G4MuonRadiativeDecayChannelWithSpin("mu+",0.014));
  G4MuonPlus::MuonPlusDefinition() -> SetDecayTable(MuonPlusDecayTable);

  G4DecayTable* MuonMinusDecayTable = new G4DecayTable();
  MuonMinusDecayTable -> Insert(new G4MuonDecayChannelWithSpin("mu-",0.986));
  MuonMinusDecayTable -> Insert(new G4MuonRadiativeDecayChannelWithSpin("mu-",0.014));
  G4MuonMinus::MuonMinusDefinition() -> SetDecayTable(MuonMinusDecayTable);


}


void muPhysicsList::ConstructBosons()
{
    // pseudo-particles
    G4Geantino::GeantinoDefinition();
    G4ChargedGeantino::ChargedGeantinoDefinition();
    
    // gamma
    // G4Gamma::Gamma(); // [yy]
    G4Gamma::GammaDefinition();
    
    // optical photon
    G4OpticalPhoton::OpticalPhotonDefinition();
}


void muPhysicsList::ConstructLeptons()
{
    // leptons
    G4Electron::ElectronDefinition();
    G4Positron::PositronDefinition();
    G4MuonPlus::MuonPlusDefinition();
    G4MuonMinus::MuonMinusDefinition();
    
    G4NeutrinoE::NeutrinoEDefinition();
    G4AntiNeutrinoE::AntiNeutrinoEDefinition();
    G4NeutrinoMu::NeutrinoMuDefinition();
    G4AntiNeutrinoMu::AntiNeutrinoMuDefinition();
}


void muPhysicsList::ConstructMesons()
{
    //  mesons
    G4PionPlus::PionPlusDefinition();
    G4PionMinus::PionMinusDefinition();
    G4PionZero::PionZeroDefinition();
    G4Eta::EtaDefinition();
    G4EtaPrime::EtaPrimeDefinition();
    G4KaonPlus::KaonPlusDefinition();
    G4KaonMinus::KaonMinusDefinition();
    G4KaonZero::KaonZeroDefinition();
    G4AntiKaonZero::AntiKaonZeroDefinition();
    G4KaonZeroLong::KaonZeroLongDefinition();
    G4KaonZeroShort::KaonZeroShortDefinition();
}


void muPhysicsList::ConstructBaryons()
{
    //  barions
    G4Proton::ProtonDefinition();
    G4AntiProton::AntiProtonDefinition();
    G4Neutron::NeutronDefinition();
    G4AntiNeutron::AntiNeutronDefinition();
}



void muPhysicsList::ConstructProcess()
{
    AddTransportation();
    ConstructEM();
    ConstructGeneral();
    ConstructDecay();  // [yy]
    
#ifdef OPT_ON
    ConstructOp();
#endif 
}

void muPhysicsList::ConstructEM()
{
    auto *theParticleIterator = GetParticleIterator();
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();
        
        if (particleName == "gamma") {
            // gamma         
            pmanager->AddDiscreteProcess(new G4PhotoElectricEffect);
            pmanager->AddDiscreteProcess(new G4ComptonScattering);
            pmanager->AddDiscreteProcess(new G4GammaConversion);
            
        } else if (particleName == "e-") {
            //electron
            pmanager->AddProcess(new G4eMultipleScattering,-1, 1,1);
            pmanager->AddProcess(new G4eIonisation,       -1, 2,2);
            pmanager->AddProcess(new G4eBremsstrahlung,   -1, 3,3);      
            
        } else if (particleName == "e+") {
            //positron
            pmanager->AddProcess(new G4eMultipleScattering, -1, 1,1);
            pmanager->AddProcess(new G4eIonisation,       -1, 2, 2);
            pmanager->AddProcess(new G4eBremsstrahlung,   -1, 3, 3);
            pmanager->AddProcess(new G4eplusAnnihilation,  0, -1, 4);
            
        } else if( particleName == "mu+" || 
                  particleName == "mu-"    ) {
            //muon  
            pmanager->AddProcess(new G4MuMultipleScattering, -1, 1,1);
            pmanager->AddProcess(new G4MuIonisation,      -1, 2, 2);
            pmanager->AddProcess(new G4MuBremsstrahlung,  -1, 3, 3);
            pmanager->AddProcess(new G4MuPairProduction,  -1, 4, 4);       
            
        } else if ((!particle->IsShortLived()) &&
                   (particle->GetPDGCharge() != 0.0) && 
                   (particle->GetParticleName() != "chargedgeantino")) {
            //all others charged particles except geantino
            pmanager->AddProcess(new G4hMultipleScattering,-1, 1,1);
            pmanager->AddProcess(new G4hIonisation,       -1, 2,2); 
        }
    }
}


#include "G4Decay.hh"

void muPhysicsList::ConstructGeneral()
{
    // Add Decay Process
    G4Decay* theDecayProcess = new G4Decay();
    G4StepLimiter* theStepLimiter = new G4StepLimiter;
    auto *theParticleIterator = GetParticleIterator();
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        if (theDecayProcess->IsApplicable(*particle)) { 
            pmanager ->AddProcess(theDecayProcess);
            // set ordering for PostStepDoIt and AtRestDoIt
            pmanager ->SetProcessOrdering(theDecayProcess, idxPostStep);
            pmanager ->SetProcessOrdering(theDecayProcess, idxAtRest);
        }
        if ((!particle->IsShortLived()) &&
            (particle->GetPDGCharge() != 0.0) &&
            (particle->GetParticleName() != "chargedgeantino")) {
            //step limit
            pmanager->AddDiscreteProcess(theStepLimiter);
        }
    }
}

#include "G4Cerenkov.hh"

void muPhysicsList::ConstructOp()
{
    G4Cerenkov* theCerenkovProcess = new G4Cerenkov("Cerenkov");
    theCerenkovProcess->SetMaxNumPhotonsPerStep(300);
    theCerenkovProcess->SetTrackSecondariesFirst(true);
    
    auto *theParticleIterator = GetParticleIterator();
    theParticleIterator->reset();
    while( (*theParticleIterator)() ){
        G4ParticleDefinition* particle = theParticleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        G4String particleName = particle->GetParticleName();
        if (theCerenkovProcess->IsApplicable(*particle)) {
            pmanager->AddProcess(theCerenkovProcess);
            pmanager->SetProcessOrdering(theCerenkovProcess,idxPostStep);
        }
    }
}

// http://geant4.lngs.infn.it/corso_infn/doxygen/task3/task3b/index.html
 void muPhysicsList::ConstructDecay()
{
  G4Decay* theDecayProcess = new G4DecayWithSpin();

  G4ParticleDefinition* muPlus = G4MuonPlus::MuonPlusDefinition();
  G4ParticleDefinition* muMinus= G4MuonMinus::MuonMinusDefinition();
 
  G4ProcessManager* muPlusManager = muPlus->GetProcessManager();
  muPlusManager->AddProcess(theDecayProcess);
  muPlusManager->SetProcessOrdering(theDecayProcess, idxPostStep);
  muPlusManager->SetProcessOrdering(theDecayProcess, idxAtRest);
 
  G4ProcessManager* muMinusManager = muMinus->GetProcessManager();
  muMinusManager->AddProcess(theDecayProcess);
  muMinusManager->SetProcessOrdering(theDecayProcess, idxPostStep);
  muMinusManager->SetProcessOrdering(theDecayProcess, idxAtRest);
 }

void muPhysicsList::SetCuts()
{
    if (verboseLevel >0){
        G4cout << "muPhysicsList::SetCuts:";
        G4cout << "CutLength : " << G4BestUnit(defaultCutValue,"Length") << G4endl;
    }
    
    // set cut values for gamma at first and for e- second and next for e+,
    // because some processes for e+/e- need cut values for gamma
    //
    SetCutValue(defaultCutValue, "gamma");
    SetCutValue(defaultCutValue, "e-");
    SetCutValue(defaultCutValue, "e+");
    
    if (verboseLevel>0) DumpCutValuesTable();
}


