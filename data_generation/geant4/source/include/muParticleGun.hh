
#ifndef muParticleGun_h
#define muParticleGun_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleGunMessenger.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include <fstream>
#include <vector>

class G4ParticleTable;
class G4Event;
class muDetectorConstraction;
class muParticleGunMessenger;


class muParticleGun : public G4ParticleGun
{
    friend class muParticleGunMessenger;
public:
    muParticleGun();
    ~muParticleGun();
    
public:
    virtual void GeneratePrimaryVertex(G4Event*);
    void GenerateNeutron(G4PrimaryParticle* neutron[1]);
    void GenerateMuon(G4PrimaryParticle* muon[1]);
    void GenerateFluxNeutron(G4PrimaryParticle* neutron[1], G4ThreeVector vPos);
    void GenerateFluxNeutronSp(G4PrimaryParticle* neutron[1], G4ThreeVector vPos);
    void GenerateFission(G4PrimaryParticle* neutron[1]);
    
    void GenerateTwoGamma(G4PrimaryParticle* gamma[2]);
    void GenerateThreeGamma(G4PrimaryParticle* gamma[3]);
    void GenerateGamma1275(G4PrimaryParticle* gamma[1]);
    void GeneratePositron(G4PrimaryParticle* positron[1]);
    
    G4double sigma(G4double x,G4double y) const {
        return (x+y-1.)*(x+y-1.)/(x*x*y*y);
    }
    
    G4double phi1(G4double x,G4double y) const {
        return std::acos((2.-2.*x -2.*y + x*y)/(x*y));
    }
    
    G4double phi2(G4double x,G4double y) const {
        return -1.*std::acos(-1.*(x*x + x*y -2.*x -2.*y + 2.)/(x*(2.-x-y)));
    }
    
    G4double beta(G4double x){
        const G4double me2=electron_mass_c2 * electron_mass_c2;
        const G4double emax=543 * keV;
        const G4double p2=(2.*electron_mass_c2+emax)*emax;
        G4double val = (std::sqrt(me2+p2)-std::sqrt(me2+x*x))*x;
        return val*val;
    }
    
    G4double cosThetaMax;
    
    const G4ThreeVector& PutCentre();
    const G4ThreeVector& PutTop();
    const G4ThreeVector& PutFlux();
    
    G4double LogLogInterpolatorCalculate(G4double);
    G4double LogLogInterpolatorCalculateSp(G4double);
    G4double LogLogInterpolatorCalculateFission(G4double);
    
private:
    std::ifstream tableFile;
    std::vector<long double> muonE;
    std::vector<long double> muonFlux;
    std::vector<long double> muonPDF;
    
    std::ifstream tableFileSp;
    std::vector<long double> spE;
    std::vector<long double> spFlux;
    std::vector<long double> spPDF;
    
    std::ifstream tableFileFission;
    std::vector<long double> fissionE;
    std::vector<long double> fissionFlux;
    std::vector<long double> fissionPDF;
    
    G4ParticleTable*					particleTable;
    const muDetectorConstraction*		muDC;
    
    G4int positionFlag;
    enum{ UserPos=0, Top, Centre,flux};
    G4int particleFlag;
    enum{ User=0, Muon, Neutron,fluxNeutron,fluxNeutronSp,fission, positron, gamma2, gamma3, gamma1275};
    G4double monoEnergy;
    
    muParticleGunMessenger*	pMessenger;
    
    
};

#endif
