
#include "muParticleGun.hh"
#include "muParticleGunMessenger.hh"
#include "muDetectorConstruction.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4RotationMatrix.hh"
#include "Randomize.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <math.h>

using CLHEP::RandFlat;

muParticleGun::muParticleGun()
:G4ParticleGun(1),
particleTable(G4ParticleTable::GetParticleTable()),
positionFlag(0),
particleFlag(0),
monoEnergy(1.0*keV),pMessenger(0)

{
    pMessenger = new muParticleGunMessenger(this);
    
    cosThetaMax = 10./17.;
    
    
    tableFile.open("muonTable.dat");
    double f1, f2;
    while(tableFile >> f1 >> f2){
        muonE.push_back(f1);
        muonFlux.push_back(f2);
    }
    
    double totalFlux = 0;
    for(decltype(muonFlux.size()) i=0;i<muonFlux.size();i++){
        totalFlux += muonFlux.at(i);
    }
    double muonFactor = 1.0 / totalFlux;
    totalFlux = 0.0;
    
    for(decltype(muonFlux.size()) i=0;i<muonFlux.size();i++){
        totalFlux += muonFlux.at(i);
        muonPDF.push_back(muonFactor* totalFlux);
    }
    
    tableFileSp.open("total_table.dat");
    //double f1, f2;
    while(tableFileSp >> f1 >> f2){
        spE.push_back(f1);
        spFlux.push_back(f2);
    }
    
    totalFlux = 0;
    for(decltype(spFlux.size()) i=0;i<spFlux.size();i++){
        totalFlux += spFlux.at(i);
    }
    double spFactor = 1.0 / totalFlux;
    totalFlux = 0.0;
    
    for(decltype(spFlux.size()) i=0;i<spFlux.size();i++){
        totalFlux += spFlux.at(i);
        spPDF.push_back(spFactor* totalFlux);
    }
    
    tableFileFission.open("cf.dat");
    //double f1, f2;
    while(tableFileFission >> f1 >> f2){
        fissionE.push_back(f1);
        fissionFlux.push_back(f2);
    }
    
    totalFlux = 0;
    for(decltype(fissionFlux.size()) i=0;i<fissionFlux.size();i++){
        totalFlux += fissionFlux.at(i);
    }
    double fissionFactor = 1.0 / totalFlux;
    totalFlux = 0.0;
    
    for(decltype(fissionFlux.size()) i=0;i<fissionFlux.size();i++){
        totalFlux += fissionFlux.at(i);
        fissionPDF.push_back(fissionFactor* totalFlux);
    }
    
}

muParticleGun::~muParticleGun()
{
    delete pMessenger;
}

void muParticleGun::GeneratePrimaryVertex(G4Event* anEvent)
{
    G4ThreeVector vPos;
    if(positionFlag == Top)				vPos = PutTop();
    else if (positionFlag == Centre)	vPos = PutCentre();
    else if (positionFlag == flux)  	vPos = PutFlux();
    else 								vPos = particle_position;
    
    G4PrimaryVertex* vertex = new G4PrimaryVertex(vPos,particle_time);
    
    if (particleFlag == Muon){
        G4PrimaryParticle* particle[1]={0};
        GenerateMuon(particle);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == Neutron){
        G4PrimaryParticle* particle[1]={0};
        GenerateNeutron(particle);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == fluxNeutron){
        G4PrimaryParticle* particle[1]={0};
        GenerateFluxNeutron(particle,vPos);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == fluxNeutronSp){
        G4PrimaryParticle* particle[1]={0};
        GenerateFluxNeutronSp(particle,vPos);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == fission){
        G4PrimaryParticle* particle[1]={0};
        GenerateFission(particle);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == positron){
        G4PrimaryParticle* particle[1]={0};
        GeneratePositron(particle);
        vertex->SetPrimary(particle[0]);
    }else if(particleFlag == gamma2){
        G4PrimaryParticle* particle[2]={0,0};
        GenerateTwoGamma(particle);
        vertex->SetPrimary(particle[0]);
        vertex->SetPrimary(particle[1]);
    }else if(particleFlag == gamma3){
        G4PrimaryParticle* particle[3]={0,0,0};
        GenerateThreeGamma(particle);
        vertex->SetPrimary(particle[0]);
        vertex->SetPrimary(particle[1]);
        vertex->SetPrimary(particle[2]);
    }else if(particleFlag == gamma1275){
        G4PrimaryParticle* particle[1]={0};
        GenerateGamma1275(particle);
        vertex->SetPrimary(particle[0]);
    }else{
        G4double mass = particle_definition->GetPDGMass();	
        for(G4int i=0; i<NumberOfParticlesToBeGenerated; i++){
            G4PrimaryParticle* particle = new G4PrimaryParticle(particle_definition);
            particle->SetKineticEnergy(particle_energy);
            particle->SetMass(mass);
            particle->SetMomentumDirection(particle_momentum_direction);
            particle->SetCharge(particle_charge);
            particle->SetPolarization(particle_polarization.x(),
                                      particle_polarization.y(),
                                      particle_polarization.z());
            vertex->SetPrimary(particle);
            
        }
    }
    anEvent->AddPrimaryVertex(vertex);
}

void muParticleGun::GenerateNeutron(G4PrimaryParticle* neutron[1])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("neutron");
    neutron[0] = new G4PrimaryParticle(pD);
    neutron[0]->SetKineticEnergy(1.0 * GeV);
    neutron[0]->SetMomentumDirection(G4ThreeVector(0,0,-1.0));
}

void muParticleGun::GenerateFluxNeutron(G4PrimaryParticle* neutron[1],G4ThreeVector vPos)
{
    G4ParticleDefinition* pD = particleTable->FindParticle("neutron");
    neutron[0] = new G4PrimaryParticle(pD);
    //Set energy
    neutron[0]->SetKineticEnergy(monoEnergy);
    
    G4double angleX = asin(2*RandFlat::shoot(0.0,1.0)-1.0);
    G4double delta = RandFlat::shoot(0.0,pi);
    
    G4ThreeVector momVec;
    momVec.setX(-1*vPos.getX()/vPos.getR());
    momVec.setY(-1*vPos.getY()/vPos.getR());
    momVec.setZ(-1*vPos.getZ()/vPos.getR());
    
    G4double theta = momVec.getTheta();
    G4double phi = momVec.getPhi();
    momVec.rotateZ(-1*phi);
    momVec.rotateY(-1*theta);
    momVec.rotateY(angleX);
    momVec.rotateZ(delta);
    momVec.rotateY(theta);
    momVec.rotateZ(phi);
    
    neutron[0]->SetMomentumDirection(momVec);
}

void muParticleGun::GenerateFluxNeutronSp(G4PrimaryParticle* neutron[1],G4ThreeVector vPos)
{
    G4ParticleDefinition* pD = particleTable->FindParticle("neutron");
    neutron[0] = new G4PrimaryParticle(pD);
    //Set energy
    G4double prob = RandFlat::shoot(0.0,1.0);
    neutron[0]->SetKineticEnergy( LogLogInterpolatorCalculateSp(prob) * MeV); //real spectrum
    
    G4double angleX = asin(2*RandFlat::shoot(0.0,1.0)-1.0);
    G4double delta = RandFlat::shoot(0.0,pi);
    
    G4ThreeVector momVec;
    momVec.setX(-1*vPos.getX()/vPos.getR());
    momVec.setY(-1*vPos.getY()/vPos.getR());
    momVec.setZ(-1*vPos.getZ()/vPos.getR());
    
    G4double theta = momVec.getTheta();
    G4double phi = momVec.getPhi();
    momVec.rotateZ(-1*phi);
    momVec.rotateY(-1*theta);
    momVec.rotateY(angleX);
    momVec.rotateZ(delta);
    momVec.rotateY(theta);
    momVec.rotateZ(phi);
    
    neutron[0]->SetMomentumDirection(momVec);
}

void muParticleGun::GenerateFission(G4PrimaryParticle* neutron[1])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("neutron");
    neutron[0] = new G4PrimaryParticle(pD);
    //Set energy
    G4double prob = RandFlat::shoot(0.0,1.0);
    neutron[0]->SetKineticEnergy( LogLogInterpolatorCalculateFission(prob) * MeV); //real spectrum
    
    G4double px, py, pz; 
    G4double cs, sn, phi;
    cs    =  RandFlat::shoot(-1.0,1.0);
    sn    =  std::sqrt((1.0-cs)*(1.0+cs));   
    phi   =  RandFlat::shoot(0., CLHEP::twopi);   
    px    =  sn*std::cos(phi);
    py    =  sn*std::sin(phi);
    pz    =  cs; 
    
    neutron[0]->SetMomentumDirection(G4ThreeVector(px, py, pz));
}
void muParticleGun::GenerateMuon(G4PrimaryParticle* muon[1])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("mu-");
    muon[0] = new G4PrimaryParticle(pD);
    muon[0]->SetMomentumDirection(G4ThreeVector(0,0,-1.0));//real shoot
    G4double prob = RandFlat::shoot(0.0,1.0);
    muon[0]->SetKineticEnergy( LogLogInterpolatorCalculate(prob) * GeV); //real spectrum
}

G4double muParticleGun::LogLogInterpolatorCalculate(G4double x){
    
    G4double value = 0;
    
    if (x > 0.9999999 || x < 0.0) return 0.0;
    
    if(x < muonPDF.at(1) || x == 0.0){
        value = 0.0;
    }else if( x > 1.0){
        value = 0.0;
    }else {
        size_t i = 0;
        for(i=0;i<muonPDF.size();i++){
            if(x < muonPDF.at(i)){ break; }
        }
        G4double e1 = muonPDF.at(i-1);
        G4double e2 = muonPDF.at(i);
        G4double d1 = muonE.at(i-1);
        G4double d2 = muonE.at(i);
        
        value = (std::log10(d1)*std::log10(e2/x) + std::log10(d2)*std::log10(x/e1)) / std::log10(e2/e1);
        value = std::pow(10.0,value);
    }
    return value;
}



G4double muParticleGun::LogLogInterpolatorCalculateSp(G4double x){
    
    G4double value = 0;
    
    if (x > 0.9999999 || x < 0.0) return 0.0;
    
    if(x < spPDF.at(1) || x == 0.0){
        value = 0.0;
    }else if( x > 1.0){
        value = 0.0;
    }else {
        size_t i = 0;
        for(i=0;i<spPDF.size();i++){
            if(x < spPDF.at(i)){ break; }
        }
        G4double e1 = spPDF.at(i-1);
        G4double e2 = spPDF.at(i);
        G4double d1 = spE.at(i-1);
        G4double d2 = spE.at(i);
        
        value = (std::log10(d1)*std::log10(e2/x) + std::log10(d2)*std::log10(x/e1)) / std::log10(e2/e1);
        value = std::pow(10.0,value);
    }
    return value;
}

G4double muParticleGun::LogLogInterpolatorCalculateFission(G4double x){
    
    G4double value = 0;
    
    if (x > 0.9999999 || x < 0.0) return 0.0;
    
    if(x < fissionPDF.at(1) || x == 0.0){
        value = 0.0;
    }else if( x > 1.0){
        value = 0.0;
    }else {
        size_t i = 0;
        for(i=0; i<fissionPDF.size();i++){
            if(x < fissionPDF.at(i)){ break; }
        }
        G4double e1 = fissionPDF.at(i-1);
        G4double e2 = fissionPDF.at(i);
        G4double d1 = fissionE.at(i-1);
        G4double d2 = fissionE.at(i);
        
        value = (std::log10(d1)*std::log10(e2/x) + std::log10(d2)*std::log10(x/e1)) / std::log10(e2/e1);
        value = std::pow(10.0,value);
    }
    return value;
}

void muParticleGun::GenerateTwoGamma(G4PrimaryParticle* gamma[2])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("gamma");
    gamma[0] = new G4PrimaryParticle(pD);
    gamma[1] = new G4PrimaryParticle(pD);
    gamma[0]->SetMass(0.);
    gamma[1]->SetMass(0.);
    gamma[0]->SetCharge(0.);
    gamma[1]->SetCharge(0.);
    gamma[0]->SetKineticEnergy(electron_mass_c2);
    gamma[1]->SetKineticEnergy(electron_mass_c2);
    
    G4double px, py, pz;
    G4double cs, sn, phi;
    cs    =  RandFlat::shoot(-1.,1.);
    sn    =  std::sqrt((1.0-cs)*(1.0+cs));
    phi   =  RandFlat::shoot(0., CLHEP::twopi);
    px    =  sn*std::cos(phi);
    py    =  sn*std::sin(phi);
    pz    =  cs;
    gamma[0]->SetMomentumDirection(G4ThreeVector(px, py, pz));
    gamma[0]->SetPolarization(G4ThreeVector(px, py, pz));
    gamma[1]->SetMomentumDirection(G4ThreeVector(-1.*px, -1.*py, -1.*pz));
    gamma[1]->SetPolarization(G4ThreeVector(px, py, pz));
}

void muParticleGun::GenerateThreeGamma(G4PrimaryParticle* gamma[3])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("gamma");
    gamma[0] = new G4PrimaryParticle(pD);
    gamma[1] = new G4PrimaryParticle(pD);
    gamma[2] = new G4PrimaryParticle(pD);
    gamma[0]->SetMass(0.);
    gamma[1]->SetCharge(0.);
    gamma[2]->SetCharge(0.);
    
    // Determine 3 gamma energy  (based on Tanioka's program)
    const G4double eps=0.001;//w1 and w2 under line
    const G4double Sigmax=1.0;//sigma max
    G4double w0 = RandFlat::shoot(eps,1.0);//energy gamma[0]
    G4double w1 = RandFlat::shoot(eps,1.0);//energy gamma[1]
    G4double w2 = 2.0-w0-w1;                      //energy gamma[2]
    G4double z  = RandFlat::shoot(0.,Sigmax);
    while ( w2 >1. || z > sigma(w0,w1) ){
        w0 = RandFlat::shoot(eps,1.0);//energy gamma[0]
        w1 = RandFlat::shoot(eps,1.0);//energy gamma[1]
        w2 = 2.0-w0-w1;                      //energy gamma[2]
        z  = RandFlat::shoot(0.,Sigmax);
    }
    
    G4double theta =  std::acos(RandFlat::shoot(-1.,1.));
    G4double phi   =  RandFlat::shoot(0., CLHEP::twopi);
    
    G4ThreeVector pMom0(1.0, 0.0, 0.0);
    pMom0.rotateY(theta);
    pMom0.rotateZ(phi);
    gamma[0]->SetKineticEnergy(electron_mass_c2*w0);
    gamma[0]->SetMomentumDirection(pMom0);
    
    G4double p1 = phi1(w0,w1); // angle between gamma[0] and gamma[1]
    G4ThreeVector pMom1(std::cos(p1), std::sin(p1), 0.0);
    pMom1.rotateY(theta);
    pMom1.rotateZ(phi);
    gamma[1]->SetKineticEnergy(electron_mass_c2*w1);
    gamma[1]->SetMomentumDirection(pMom1);
    
    G4double p2 = phi2(w0,w1); // angle between gamma[0] and gamma[2]
    G4ThreeVector pMom2(std::cos(p2), std::sin(p2), 0.0);
    pMom2.rotateY(theta);
    pMom2.rotateZ(phi);
    gamma[2]->SetKineticEnergy(electron_mass_c2*w2);
    gamma[2]->SetMomentumDirection(pMom2);
    
    //G4cout << w0 << "( " << w0 <<","<< w0*0.<< ")" << G4endl;
    //G4cout << w1 << "( " << w1*std::cos(p1) <<","<< w1*std::sin(p1) << ")" << G4endl;
    //G4cout << w2 << "( " << w2*std::cos(p2) <<","<< w2*std::sin(p2) << ")" << G4endl;
    //G4cout << "(" << theta <<","<< phi <<")"<< G4endl;
    
}


void muParticleGun::GenerateGamma1275(G4PrimaryParticle* gamma[1])
{
    G4ParticleDefinition* pD = particleTable->FindParticle("gamma");
    gamma[0] = new G4PrimaryParticle(pD);
    gamma[0]->SetMass(0.);
    gamma[0]->SetCharge(0.);
    gamma[0]->SetKineticEnergy(1274.6*keV);
    G4double px, py, pz;
    G4double cs, sn, phi;
    cs    =  RandFlat::shoot(-1.,0.);  // only downward
    sn    =  std::sqrt((1.0-cs)*(1.0+cs));
    phi   =  RandFlat::shoot(0., CLHEP::twopi);
    px    =  sn*std::cos(phi);
    py    =  sn*std::sin(phi);
    pz    =  cs;
    gamma[0]->SetMomentumDirection(G4ThreeVector(px, py, pz));
    
}

void muParticleGun::GeneratePositron(G4PrimaryParticle* positron[1])
{
    G4cout <<"GeneratePositron : cosThetaMax=" << cosThetaMax << G4endl;
    
    G4ParticleDefinition* pD = particleTable->FindParticle("e+");
    positron[0] = new G4PrimaryParticle(pD);
    positron[0]->SetMass(electron_mass_c2);
    positron[0]->SetCharge(1.);
    G4double px, py, pz;
    G4double cs, sn, phi;
    cs    =  RandFlat::shoot(-1.,cosThetaMax);  // only upward
    sn    =  std::sqrt((1.0-cs)*(1.0+cs));
    phi   =  RandFlat::shoot(0., CLHEP::twopi);
    px    =  sn*std::cos(phi);
    py    =  sn*std::sin(phi);
    pz    =  cs;
    positron[0]->SetMomentumDirection(G4ThreeVector(px, py, -1*pz)); // up
    
    const G4double emax = 0.543 * MeV;
    const G4double pmax = std::sqrt((2*electron_mass_c2+emax)*emax);
    G4double pep=RandFlat::shoot(0.,pmax );
    G4double prob = RandFlat::shoot(0.,1.5*beta(pmax*0.5));
    while (  prob > beta(pep) ){
        pep=RandFlat::shoot(0.,pmax);
        prob = RandFlat::shoot(0.,1.5*beta(pmax*0.5));
    }
    positron[0]->SetKineticEnergy(std::sqrt(electron_mass_c2*electron_mass_c2+pep*pep)-electron_mass_c2); // kinetic energy
}



const G4ThreeVector& muParticleGun::PutCentre(){
    static G4ThreeVector vPos(0.0,0.0,0.0);
    return vPos;
}

const G4ThreeVector& muParticleGun::PutTop(){
    static G4ThreeVector vPos(0.0,0.0,49.9*cm);
    return vPos;
}
const G4ThreeVector& muParticleGun::PutFlux(){
    G4double radius = 40*cm;
    G4double phi = RandFlat::shoot(0.0,twopi);
    G4double theta = RandFlat::shoot(0.0,pi);
    G4double posX =radius*sin(theta)*cos(phi);
    G4double posY =radius*sin(theta)*sin(phi);
    G4double posZ =radius*cos(theta);
    static G4ThreeVector vPos(0,0,0);
    vPos.setX(posX);
    vPos.setY(posY);
    vPos.setZ(posZ);
    return vPos;
}
